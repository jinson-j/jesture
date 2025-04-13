import pyray as rl

# Text wrapping functions
def wrap_text(text, font_size, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + " " + word if current else word
        if rl.measure_text(test.encode('utf-8'), font_size) <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def get_fitting_font_size(text, box_width, box_height, max_size, line_spacing=4):
    font_size = max_size
    while font_size > 0:
        lines = wrap_text(text, font_size, box_width)
        total_height = len(lines) * font_size + (len(lines) - 1) * line_spacing
        if total_height <= box_height:
            return font_size, lines
        font_size -= 1
    return 1, wrap_text(text, 1, box_width)

# Common drawing functions
def calculate_distance(a, b):
    """Calculate distance between two points"""
    return ((b[0] - a[0])**2 + (b[1] - a[1])**2)**0.5
