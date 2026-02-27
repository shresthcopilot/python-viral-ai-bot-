import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ================= CONFIG =================
WIDTH, HEIGHT = 1280, 720
TOOLBAR_HEIGHT = 100
MAX_POINTS = 8

COLORS = [
    (255, 0, 255),   # Purple
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 255, 255),   # Yellow
    (0, 0, 0)        # Eraser
]

# ================= HAND SETUP =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    brush_color = COLORS[0]
    brush_thickness = 10

    points = deque(maxlen=MAX_POINTS)

    prev_time = 0
    last_clear_time = 0
    CLEAR_COOLDOWN = 1.5

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ===== Toolbar =====
        cv2.rectangle(frame, (0, 0), (w, TOOLBAR_HEIGHT), (30, 30, 30), -1)

        # Color buttons
        for i, color in enumerate(COLORS):
            cx = 100 + i * 100
            cv2.circle(frame, (cx, 50), 30, color, -1)

        # CLEAR Button
        cv2.rectangle(frame, (900, 20), (1050, 80), (80, 80, 80), -1)
        cv2.putText(frame, "CLEAR", (915, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                lm = hand_landmarks.landmark
                fingers = []

                # Thumb
                fingers.append(1 if lm[4].x > lm[3].x else 0)

                # Other fingers
                for i in range(1, 5):
                    fingers.append(
                        1 if lm[tip_ids[i]].y <
                        lm[tip_ids[i] - 2].y else 0
                    )

                x = int(lm[8].x * w)
                y = int(lm[8].y * h)

                # ===== Pinch → Brush Size Control =====
                x1 = int(lm[4].x * w)
                y1 = int(lm[4].y * h)
                distance = np.hypot(x - x1, y - y1)

                if fingers[0] and fingers[1] and sum(fingers) == 2:
                    brush_thickness = int(np.interp(distance, [20, 200], [5, 50]))
                    cv2.line(frame, (x, y), (x1, y1), (255, 255, 255), 2)

                # ===== Draw Mode =====
                elif fingers[1] and sum(fingers) == 1:
                    points.append((x, y))

                    avg_x = int(np.mean([p[0] for p in points]))
                    avg_y = int(np.mean([p[1] for p in points]))

                    cv2.circle(frame, (avg_x, avg_y),
                               brush_thickness // 2,
                               brush_color, -1)

                    if len(points) > 1:
                        cv2.line(canvas,
                                 points[-2],
                                 points[-1],
                                 brush_color,
                                 brush_thickness)

                else:
                    points.clear()

                # ===== Selection Mode (Index + Middle) =====
                if fingers[1] and fingers[2]:
                    for i in range(len(COLORS)):
                        cx = 100 + i * 100
                        if abs(x - cx) < 40 and y < TOOLBAR_HEIGHT:
                            brush_color = COLORS[i]

                    # Click CLEAR button
                    if 900 < x < 1050 and 20 < y < 80:
                        current_time = time.time()
                        if current_time - last_clear_time > CLEAR_COOLDOWN:
                            canvas = np.zeros_like(canvas)
                            last_clear_time = current_time
                            print("🗑 Canvas Cleared")

                # ===== Open Palm Clear (All fingers up) =====
                if sum(fingers) == 5:
                    current_time = time.time()
                    if current_time - last_clear_time > CLEAR_COOLDOWN:
                        canvas = np.zeros_like(canvas)
                        last_clear_time = current_time
                        print("🖐 Open Palm Clear")

                mp_draw.draw_landmarks(frame,
                                       hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)

        # ===== Blend Canvas =====
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

        final = cv2.add(frame_bg, canvas_fg)

        # ===== FPS =====
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(final, f"FPS: {int(fps)}",
                    (20, 680),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.putText(final, f"Brush: {brush_thickness}",
                    (1050, 680),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        cv2.imshow("Air Paint Pro v2", final)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = np.zeros_like(canvas)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()