import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import pyray as rl
import math
from google import genai
import PIL.Image
from utilities import wrap_text, get_fitting_font_size

# Initialize Gemini API
client = genai.Client(api_key='INPUT KEY HERE')

# MediaPipe setup
model_path = "gesture_recognizer.task" 
full_drawing_path = "screenshot.png"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Game states
class GameState:
    WAITING = 0
    DRAWING = 1
    JUDGING = 2

# Game variables
current_game_state = GameState.WAITING
start_time = 0
drawing_time = 30
current_prompt = ""
results_text = ""

# Hand tracking variables
player1_position = [200, 300]
player2_position = [800, 300]
player1_is_fist = False
player2_is_fist = False
player1_has_hand = False
player2_has_hand = False
player1_prev_pos = None
player2_prev_pos = None
unassigned_hands = []

# Display settings
frame_width = 640
frame_height = 480
window_width = 1200
window_height = 900
player1_bounds = (0, 0, window_width // 2, window_height)
player2_bounds = (window_width // 2, 0, window_width // 2, window_height)
brush_size = 5.0

def generate_prompt():
    global current_prompt
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Generate a drawing prompt. Be creative. Keep the prompt short and simple."])
    current_prompt = response.text.strip()

def grade_drawings():
    image = PIL.Image.open(full_drawing_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            "You are judging a drawing contest between two players. The left half of the image is Player 1's drawing and the right half is Player 2's drawing. " +
            f"The prompt was: '{current_prompt}'. Compare both drawings, score each from 1-10, and declare a winner. Be concise and friendly. Both players are adults", 
            image])
    
    return response.text

def clear_canvas():
    global canvas
    rl.begin_texture_mode(canvas)
    rl.clear_background(rl.WHITE)

    # Draw dividing line
    rl.draw_line(window_width // 2, 0, window_width // 2, window_height, rl.BLACK)
    
    # Draw player labels
    rl.draw_text("PLAYER 1", 10, window_height - 30, 20, rl.BLACK)
    rl.draw_text("PLAYER 2", window_width - 112, window_height - 30, 20, rl.BLACK)
    
    rl.end_texture_mode()

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore 
    global player1_position, player2_position, player1_is_fist, player2_is_fist
    global player1_has_hand, player2_has_hand, unassigned_hands
    
    # Reset hands collection
    unassigned_hands = []
    
    # Process hand landmarks if available
    if result.hand_landmarks:
        # Reset hand states
        player1_has_hand = False
        player2_has_hand = False
        player1_is_fist = False
        player2_is_fist = False
        
        # Collect all hands
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            x = int(hand_landmarks[0].x * window_width)
            y = int(hand_landmarks[0].y * window_height)
            
            # Get gesture if available
            is_fist = False
            if result.gestures and i < len(result.gestures):
                gesture_info = result.gestures[i][0]
                is_fist = (gesture_info.category_name == "Closed_Fist")
            
            # Store hand information
            unassigned_hands.append({
                'position': [x, y],
                'is_fist': is_fist
            })
    
    # Assign hands to players based on position
    left_hands = []
    right_hands = []
    
    for hand in unassigned_hands:
        if hand['position'][0] < window_width // 2:
            left_hands.append(hand)
        else:
            right_hands.append(hand)
    
    # Assign hand to player 1
    if left_hands:
        player1_has_hand = True
        player1_position = left_hands[0]['position']
        player1_is_fist = left_hands[0]['is_fist']
    
    # Assign hand to player 2
    if right_hands:
        player2_has_hand = True
        player2_position = right_hands[0]['position']
        player2_is_fist = right_hands[0]['is_fist']

# Main program
def run_multiplayer_mode(drawing_time=30):
    global canvas, player1_prev_pos, player2_prev_pos, current_game_state, start_time, results_text
    
    # Update global drawing time
    globals()['drawing_time'] = drawing_time
    
    # Initialize MediaPipe
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2)
    
    # Initialize Raylib
    rl.init_window(window_width, window_height, "TWO-PLAYER DRAWING BATTLE!")
    rl.set_target_fps(60)
    
    # Create canvas
    canvas = rl.load_render_texture(window_width, window_height)
    clear_canvas()
    
    with GestureRecognizer.create_from_options(options) as recognizer:
        # Set up webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # Create frame buffer
        pixel_count = frame_width * frame_height * 3
        blank_data = rl.ffi.new("unsigned char[]", pixel_count)
        
        # Set up image
        image = rl.ffi.new("Image *")
        image.data = blank_data
        image.width = frame_width
        image.height = frame_height
        image.mipmaps = 1
        image.format = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8
        
        frame_texture = rl.load_texture_from_image(image[0])
        
        init_time = time.time()
        
        # Main game loop
        while not rl.window_should_close():
            # Read camera frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Send frame to MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int((time.time() - init_time) * 1000)
            recognizer.recognize_async(mp_image, frame_timestamp_ms)
            
            # Update texture
            frame_data = rgb_frame.astype(np.uint8).tobytes()
            frame_data_ptr = rl.ffi.new("unsigned char[]", frame_data)
            rl.update_texture(frame_texture, frame_data_ptr)
            
            # Handle keyboard input
            if rl.is_key_pressed(rl.KEY_SPACE) and current_game_state == GameState.WAITING:
                generate_prompt()
                current_game_state = GameState.DRAWING
                start_time = time.time()
                clear_canvas()
            elif rl.is_key_pressed(rl.KEY_R) and current_game_state == GameState.JUDGING:
                current_game_state = GameState.WAITING
            elif rl.is_key_pressed(rl.KEY_Q):
                break
            
            # Game state logic
            current_time = time.time()
            if current_game_state == GameState.DRAWING:
                time_left = drawing_time - (current_time - start_time)
                
                # Check if time's up
                if time_left <= 0:
                    current_game_state = GameState.JUDGING
                    rl.take_screenshot(full_drawing_path)
                    results_text = grade_drawings()
                
                # Player 1 drawing
                if player1_has_hand and player1_is_fist:
                    if player1_prev_pos:
                        x1 = min(player1_prev_pos[0], player1_bounds[2])
                        x2 = min(player1_position[0], player1_bounds[2])
                        
                        rl.begin_texture_mode(canvas)
                        rl.draw_line_ex(
                            rl.Vector2(x1, player1_prev_pos[1]),
                            rl.Vector2(x2, player1_position[1]),
                            brush_size,
                            rl.BLACK
                        )
                        rl.end_texture_mode()
                    player1_prev_pos = player1_position.copy()
                else:
                    player1_prev_pos = None
                
                # Player 2 drawing
                if player2_has_hand and player2_is_fist:
                    if player2_prev_pos:
                        x1 = max(player2_prev_pos[0], player1_bounds[2])
                        x2 = max(player2_position[0], player1_bounds[2])
                        
                        rl.begin_texture_mode(canvas)
                        rl.draw_line_ex(
                            rl.Vector2(x1, player2_prev_pos[1]),
                            rl.Vector2(x2, player2_position[1]),
                            brush_size,
                            rl.BLACK
                        )
                        rl.end_texture_mode()
                    player2_prev_pos = player2_position.copy()
                else:
                    player2_prev_pos = None
            
            # Render UI
            rl.begin_drawing()
            rl.clear_background(rl.RAYWHITE)
            
            # Draw canvas
            rl.draw_texture_rec(
                canvas.texture,
                rl.Rectangle(0, 0, canvas.texture.width, -canvas.texture.height),
                rl.Vector2(0, 0),
                rl.WHITE
            )
            
            # Draw camera view in the middle bottom
            scale = 0.25
            camera_width = frame_width * scale
            camera_height = frame_height * scale
            rl.draw_texture_ex(
                frame_texture,
                rl.Vector2(window_width/2 - camera_width/2, window_height - camera_height - 10),
                0,
                scale,
                rl.WHITE
            )
            
            # Draw player cursors
            if player1_has_hand:
                cursor_color = rl.RED if player1_is_fist else rl.BLUE
                rl.draw_circle(player1_position[0], player1_position[1], brush_size, cursor_color)
            
            if player2_has_hand:
                cursor_color = rl.RED if player2_is_fist else rl.BLUE
                rl.draw_circle(player2_position[0], player2_position[1], brush_size, cursor_color)
            
            # Game state UI
            if current_game_state == GameState.WAITING:
                rl.draw_rectangle(0, 0, window_width, 90, rl.LIGHTGRAY)
                rl.draw_text("Press SPACE to start a new round!", window_width//2 - 460, 20, 50, rl.BLACK)
            elif current_game_state == GameState.DRAWING:
                rl.draw_rectangle(0, 0, window_width, 90, rl.LIGHTGRAY)
                time_left = max(0, drawing_time - (current_time - start_time))
                rl.draw_text(f"{current_prompt}", 20, 10, 30, rl.BLACK)
                rl.draw_text(f"{int(time_left)}", window_width - 100, 20, 50, rl.RED if time_left < 10 else rl.BLACK)
            elif current_game_state == GameState.JUDGING:
                
                
                # Show results
                result_box_width = window_width - 40
                font_size, wrapped_lines = get_fitting_font_size(results_text, result_box_width, 400, 24)
                
                
                rl.draw_rectangle(0, 0, window_width, 90 + 40 * len(wrapped_lines), rl.LIGHTGRAY)
                rl.draw_text("RESULTS", window_width//2 - 90, 10, 40, rl.BLACK)
                rl.draw_text("Press R to play again!", window_width//2 - 100, 60, 20, rl.BLACK)

                y = 100
                for line in wrapped_lines:
                    rl.draw_text(line.encode('utf-8'), 20, y, font_size, rl.BLACK)
                    y += font_size + 4
            
            rl.end_drawing()
        
        # Cleanup
        cap.release()
        rl.unload_texture(frame_texture)
        rl.unload_render_texture(canvas)
        rl.close_window()

if __name__ == "__main__":
    run_multiplayer_mode()
