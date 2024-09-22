import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize a blank canvas
canvas = None

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Variables for tracking previous finger position
prev_x, prev_y = 0, 0

# Pen and eraser properties
pen_color = (255, 0, 0)  # Blue color for pen
eraser_color = (0, 0, 0)  # Black color for eraser (can be white if background is white)
pen_thickness = 5
eraser_thickness = 50

# Current mode (starts in pen mode)
mode = "pen"

def is_fist_closed(hand_landmarks):
    """Check if the hand is forming a fist by analyzing the relative positions of fingertips."""
    # Finger tips landmarks: Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Check if the tips of the fingers are curled towards the palm (i.e., y-coordinates are close)
    if (thumb_tip.y > index_tip.y and middle_tip.y > index_tip.y and
        ring_tip.y > index_tip.y and pinky_tip.y > index_tip.y):
        return True
    return False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (for Mediapipe processing)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    # Initialize the canvas on first frame
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger tip coordinates (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, c = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Check if the hand is making a fist (for erasing)
            if is_fist_closed(hand_landmarks):
                mode = "eraser"
            else:
                mode = "pen"

            # If the previous coordinates are (0, 0), set them to current
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Draw a line on the canvas between previous and current positions
            if mode == "pen":
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), pen_color, pen_thickness)
            elif mode == "eraser":
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), eraser_color, eraser_thickness)

            # Update previous coordinates
            prev_x, prev_y = cx, cy

            # Draw hand landmarks on the frame (optional, for visualization)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        # Reset previous coordinates if no hand is detected
        prev_x, prev_y = 0, 0

    # Merge the canvas with the original frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show the result
    cv2.imshow("Virtual Whiteboard", combined)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
