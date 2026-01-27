import cv2 as cv
import statistics
from collections import deque
from src.predict import ASLClassifier


def main():
    try:
        model = ASLClassifier()
    except Exception as e:
        print(e)
        return

    prediction_history = deque(maxlen=5)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame (stream end?). Exiting ...")
            break

        frame = cv.flip(frame, 1)

        x1, y1, x2, y2 = 350, 50, 600, 300
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        roi_crop = frame[y1:y2, x1:x2]

        if roi_crop.size != 0:
            label, confidence = model.classify_frame(roi_crop)

            prediction_history.append(label)

            try:
                stable_level = statistics.mode(prediction_history)

            except statistics.StatisticsError:
                stable_level = label

            color = (0, 255, 0) if confidence > 0.75 else (0, 0, 255)

            if confidence > 0.5:
                text = f"label: {stable_level}, conf: {confidence:.2f}"
            else:
                text = "..."

            cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv.imshow("ASL Translator", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
