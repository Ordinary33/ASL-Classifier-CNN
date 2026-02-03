import cv2 as cv
import mediapipe as mp
import statistics
from collections import deque
from src.predict import ASLClassifier

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_bbox_coordinates(landmarks, image_shape):
    h, w, c = image_shape
    x_max, y_max = 0, 0
    x_min, y_min = w, h

    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y

    offset = 40
    y_min -= offset
    y_max += offset
    x_min -= offset
    x_max += offset

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w > box_h:
        diff = (box_w - box_h) // 2
        y_min -= diff
        y_max += diff
    else:
        diff = (box_h - box_w) // 2
        x_min -= diff
        x_max += diff

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    return x_min, y_min, x_max, y_max


def main():
    try:
        model = ASLClassifier()
    except Exception as e:
        print(e)
        return

    prediction_history = deque(maxlen=5)

    string = ""
    stability_count = 0
    last_label = ""
    last_text = ""
    stability_threshold = 30

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Cannot receive frame (stream end?). Exiting ...")
                break

            frame = cv.flip(frame, 1)
            h, w, c = frame.shape
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    x1, y1, x2, y2 = get_bbox_coordinates(hand_landmarks, (h, w, c))

                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    roi_crop = frame[y1:y2, x1:x2]

                    if roi_crop.size != 0:
                        label, confidence = model.classify_frame(roi_crop)

                        prediction_history.append(label)

                        try:
                            stable_level = statistics.mode(prediction_history)

                        except statistics.StatisticsError:
                            stable_level = label

                        if stable_level == last_label:
                            stability_count += 1

                        else:
                            stability_count = 0
                            last_label = stable_level
                            last_text = None

                        if (
                            stability_count >= stability_threshold
                            and stable_level != last_text
                        ):
                            if stable_level == "space":
                                string += " "
                                last_text = stable_level
                            elif stable_level == "del":
                                string = string[:-1]
                                last_text = stable_level
                            else:
                                string += stable_level
                                last_text = stable_level

                        ratio = min(1.0, stability_count / stability_threshold)
                        prog_bar_width = (ratio) * (x2 - x1)
                        cv.rectangle(
                            frame,
                            (x1, y2 + 10),
                            (x1 + int(prog_bar_width), y2 + 30),
                            (0, 255, 0),
                            -1,
                        )
                        cv.rectangle(
                            frame, (x1, y2 + 10), (x2, y2 + 30), (255, 255, 255), 2
                        )

                        color = (0, 255, 0) if confidence > 0.75 else (0, 0, 255)

                        if confidence > 0.5:
                            text = f"label: {stable_level}, conf: {confidence:.2f}"
                        else:
                            text = "..."

                        cv.putText(
                            frame,
                            text,
                            (x1, y1 - 10),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            2,
                        )

                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            cv.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv.putText(
                frame,
                "Translation: ",
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv.putText(
                frame,
                string,
                (20, 65),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv.imshow("ASL Translator", frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
