import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(3, 1280)
cap.set(4, 720)

canvas = None
prev_x, prev_y = 0, 0

brush_color = (255, 0, 255)
brush_thickness = 8

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Draw top toolbar
    cv2.rectangle(frame, (0, 0), (w, 100), (50, 50, 50), -1)

    # Color buttons
    colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 0)]
    for i, color in enumerate(colors):
        cv2.circle(frame, (100 + i * 100, 50), 30, color, -1)

    cv2.putText(frame, "CLEAR", (650, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "SAVE", (850, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            x = int(landmarks[8].x * w)
            y = int(landmarks[8].y * h)

            fingers = []

            # Thumb
            fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

            # Other fingers
            for i in range(1, 5):
                fingers.append(1 if landmarks[tip_ids[i]].y < landmarks[tip_ids[i]-2].y else 0)

            # 🎨 SELECTION MODE (Index + Middle up)
            if fingers[1] == 1 and fingers[2] == 1:

                prev_x, prev_y = 0, 0

                # Color selection
                for i in range(len(colors)):
                    cx = 100 + i * 100
                    if abs(x - cx) < 40 and y < 100:
                        brush_color = colors[i]

                # Clear
                if 600 < x < 750 and y < 100:
                    canvas = np.zeros_like(frame)

                # Save
                if 800 < x < 950 and y < 100:
                    cv2.imwrite("air_paint.png", canvas)
                    print("Image Saved!")

            # ✍ DRAW MODE (Only index up)
            elif fingers[1] == 1 and sum(fingers) == 1:

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), brush_color, brush_thickness)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    combined = cv2.add(frame, canvas)

    cv2.imshow("Air Paint Pro", combined)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('+'):
        brush_thickness += 2

    if key == ord('-'):
        brush_thickness = max(2, brush_thickness - 2)

cap.release()
cv2.destroyAllWindows()
hands.close()
