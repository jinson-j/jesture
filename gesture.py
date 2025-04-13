import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import pyray as rl
import math
from google import genai
from google.genai import types
import PIL.Image
from utilities import wrap_text, get_fitting_font_size, calculate_distance

client = genai.Client(api_key='INPUT KEY HERE')

model_path = "gesture_recognizer.task" 
image_path = "screenshot.png"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

gesture = ""

# Raylib Defaults
ball_position = [400, 300] 
is_fist = False
has_hand = False

# Resolution setup
frame_width = 640
frame_height = 480
window_width = 1200
window_height = 900

# Paint program variables (from Go code)
class DrawMode:
    FREE = 0
    LINE = 1
    CIRCLE = 2

current_mode = DrawMode.FREE
current_color = rl.BLACK
brush_size = 5.0
canvas = None
robot_texture = None
start_pos = None
end_pos = None
is_drawing = False

guess = "I am excited to guess! :)"

def predict_drawing():
    global guess
    image = PIL.Image.open(image_path)
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "IGNORE EVERYTHING WITH A GRAY BACKGROUND! ROLL PLAY AS FRIENDLY ROBOT THAT TRIES TO GUESS THE USERS DRAWING! What am I trying to draw? Write a short sentence!", image])
    
    guess = response.text
    # print(guess)

def clear():
    global canvas, robot_texture
    rl.begin_texture_mode(canvas)
    rl.clear_background(rl.WHITE)

    rl.draw_rectangle(0, 0, window_width, 140, rl.LIGHTGRAY)

    # draw the robot png
    robot_texture = rl.load_texture("robot.png")
    rl.draw_texture_ex(robot_texture, rl.Vector2(window_width - 325, 10), 0,  0.5, rl.WHITE)

    rl.end_texture_mode()

# Callback function to process gesture recognition results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore 
    global ball_position, is_fist, has_hand, gesture, is_drawing, start_pos, end_pos

    has_hand = False
    is_fist = False

    # Hand position
    if result.hand_landmarks:
        has_hand = True
        # Get first hand landmarks
        hand_landmarks = result.hand_landmarks[0]

        # Use index finger tip (landmark 8) for painting
        x = int((hand_landmarks[0].x) * window_width)
        y = int(hand_landmarks[0].y * window_height)

        ball_position = [x, y]

    # Hand fist
    if result.gestures and result.hand_landmarks:
        gesture_info = result.gestures[0][0]
        category = gesture_info.category_name
        gesture = category
        if category == "Closed_Fist":
            is_fist = True
            if not is_drawing:
                is_drawing = True
                start_pos = ball_position.copy()
        else:
            if is_drawing and current_mode != DrawMode.FREE:
                is_drawing = False
                end_pos = ball_position.copy()

def run_singleplayer_mode():
    global canvas, ball_position, is_fist, has_hand, gesture, is_drawing, start_pos, end_pos, current_mode, current_color, brush_size, guess, robot_texture
    
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    # Initialize Raylib
    rl.init_window(window_width, window_height, "THE SMARTEST AND FRIENDLIEST ROBOT IN THE WORLD!")
    rl.set_target_fps(60)

    # Create a canvas for drawing (similar to the Go code)
    canvas = rl.load_render_texture(window_width, window_height)
    clear()

    with GestureRecognizer.create_from_options(options) as recognizer:
        # Use OpenCV's VideoCapture to start capturing from the webcam.
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows
        if not cap.isOpened():
            print("Error: Could not open webcam")
            exit()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        # Create blank RGB buffer (3 bytes per pixel)
        pixel_count = frame_width * frame_height * 3
        blank_data = rl.ffi.new("unsigned char[]", pixel_count)

        # Create and assign Image struct
        image = rl.ffi.new("Image *")
        image.data = blank_data
        image.width = frame_width
        image.height = frame_height
        image.mipmaps = 1
        image.format = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8

        frame_texture = rl.load_texture_from_image(image[0])

        prev_pos = None
        start_time = time.time()
        
        # Main loop
        while not rl.window_should_close():
            # Read camera frame
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # Update camera texture
            frame_data = rgb_frame.astype(np.uint8).tobytes()
            frame_data_ptr = rl.ffi.new("unsigned char[]", frame_data)
            rl.update_texture(frame_texture, frame_data_ptr)

            # Handle keyboard input for mode and color changes (like the Go code)
            if rl.is_key_pressed(rl.KEY_F):
                current_mode = DrawMode.FREE
            elif rl.is_key_pressed(rl.KEY_L):
                current_mode = DrawMode.LINE
            elif rl.is_key_pressed(rl.KEY_C):
                current_mode = DrawMode.CIRCLE
            elif rl.is_key_pressed(rl.KEY_X):
                clear()         
            elif rl.is_key_pressed(rl.KEY_G):
                # Save canvas to file
                rl.take_screenshot(image_path)
                predict_drawing()                
            elif rl.is_key_pressed(rl.KEY_Q):
                break

            # Color selection keys
            if rl.is_key_pressed(rl.KEY_ONE):
                current_color = rl.BLACK
            elif rl.is_key_pressed(rl.KEY_TWO):
                current_color = rl.RED
            elif rl.is_key_pressed(rl.KEY_THREE):
                current_color = rl.BLUE
            elif rl.is_key_pressed(rl.KEY_FOUR):
                current_color = rl.GREEN
                
            # Brush size adjustment
            if rl.is_key_pressed(rl.KEY_UP):
                brush_size += 1.0
            elif rl.is_key_pressed(rl.KEY_DOWN):
                brush_size = max(1.0, brush_size - 1.0)

            # Drawing logic based on mode
            if has_hand:
                if current_mode == DrawMode.FREE and is_fist:
                    if prev_pos:
                        # Draw line on canvas
                        rl.begin_texture_mode(canvas)
                        rl.draw_line_ex(
                            rl.Vector2(prev_pos[0], prev_pos[1]),
                            rl.Vector2(ball_position[0], ball_position[1]),
                            brush_size,
                            current_color
                        )
                        rl.end_texture_mode()
                    prev_pos = ball_position.copy()
                elif current_mode == DrawMode.LINE:
                    if not is_drawing and end_pos and start_pos:
                        # Draw line when released
                        rl.begin_texture_mode(canvas)
                        rl.draw_line_ex(
                            rl.Vector2(start_pos[0], start_pos[1]),
                            rl.Vector2(end_pos[0], end_pos[1]),
                            brush_size,
                            current_color
                        )
                        rl.end_texture_mode()
                        start_pos = None
                        end_pos = None
                elif current_mode == DrawMode.CIRCLE:
                    if not is_drawing and end_pos and start_pos:
                        # Draw circle when released
                        rl.begin_texture_mode(canvas)
                        radius = calculate_distance(start_pos, end_pos)

                        thickness = max(1, brush_size / 2)
                        for i in range(int(thickness)):
                            rl.draw_circle_lines(
                                int(start_pos[0]), int(start_pos[1]),
                                radius - i, current_color
                            )
                        rl.end_texture_mode()
                        start_pos = None
                        end_pos = None
            
            if not is_fist:
                prev_pos = None
            
            # Begin drawing UI
            rl.begin_drawing()
            rl.clear_background(rl.RAYWHITE)

            # Draw the canvas
            rl.draw_texture_rec(
                canvas.texture,
                rl.Rectangle(0, 0, canvas.texture.width, -canvas.texture.height),
                rl.Vector2(0, 0),
                rl.WHITE
            )

            # Draw camera view in corner
            scale = 0.25  # Show camera at 25% size
            rl.draw_texture_ex(frame_texture, rl.Vector2(window_width - frame_width * scale - 5, 10), 0, scale, rl.WHITE)
            
            # If currently creating a shape, preview it
            if is_drawing and has_hand:
                if current_mode == DrawMode.LINE:
                    rl.draw_line_ex(
                        rl.Vector2(start_pos[0], start_pos[1]),
                        rl.Vector2(ball_position[0], ball_position[1]),
                        brush_size,
                        rl.fade(current_color, 0.6)  # Semi-transparent preview
                    )
                elif current_mode == DrawMode.CIRCLE:
                    radius = calculate_distance(start_pos, ball_position)
                    rl.draw_circle_lines(
                        int(start_pos[0]), int(start_pos[1]),
                        radius, rl.fade(current_color, 0.6)
                    )

            # Draw UI indicators
            mode_text = "Free Draw" if current_mode == DrawMode.FREE else "Line Draw" if current_mode == DrawMode.LINE else "Circle Draw"
            rl.draw_text(f"Mode: {mode_text}", 10, 10, 20, rl.BLACK)
            
            # Display color name in the actual color!
            color_name = "BLACK"
            if current_color == rl.RED:
                color_name = "RED"
            elif current_color == rl.BLUE:
                color_name = "BLUE"
            elif current_color == rl.GREEN:
                color_name = "GREEN"

            # Green is hard to read
            display_color = current_color
            if current_color == rl.GREEN:
                display_color = rl.DARKGREEN
                
            rl.draw_text("Color (1-4): ", 10, 35, 20, rl.BLACK)
            rl.draw_text(color_name, 130, 35, 20, display_color)
            
            rl.draw_text(f"Brush Size: {brush_size} (Up/Down)", 10, 60, 20, rl.BLACK)
            rl.draw_text(f"Gesture: {gesture}", 10, 85, 20, rl.BLACK)
            rl.draw_text(f"\"F\" = Free, \"L\" = Line, \"C\" = Circle, \"X\" = clear, \"Q\" = quit, \"G\" = guess,", 10, 110, 20, rl.BLACK)

            # Display AI guess in a box with proper text wrapping
            guess_box_x = window_width-675
            guess_box_y = 10
            guess_box_width = 350
            guess_box_height = 100
            max_font_size = 20
            line_spacing = 4
            
            '''
            # Draw the box
            rl.draw_rectangle(guess_box_x, guess_box_y, guess_box_width, guess_box_height, rl.LIGHTGRAY)
            rl.draw_rectangle_lines(guess_box_x, guess_box_y, guess_box_width, guess_box_height, rl.DARKGRAY)
            '''

            # Fit and draw the text
            font_size, wrapped_lines = get_fitting_font_size(guess, guess_box_width - 10, guess_box_height - 10, max_font_size)
            
            y = guess_box_y + 5
            for line in wrapped_lines:
                rl.draw_text(line.encode('utf-8'), guess_box_x + 5, y, font_size, rl.BLACK)
                y += font_size + line_spacing

            # Draw cursor
            color = rl.GRAY
            if has_hand:
                color = rl.RED if is_fist else rl.BLUE
            rl.draw_circle(ball_position[0], ball_position[1], max(5, brush_size/2), color)

            rl.end_drawing()

            if rl.is_key_pressed(rl.KEY_Q):
                break
            
        # Clean up
        cap.release()
        rl.unload_texture(frame_texture)
        rl.unload_texture(robot_texture)
        rl.unload_render_texture(canvas)
        rl.close_window()

if __name__ == "__main__":
    run_singleplayer_mode()

