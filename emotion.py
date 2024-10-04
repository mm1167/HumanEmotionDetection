from fer import FER
import cv2

# Create FER detector object
detector = FER()

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect emotions
    result = detector.detect_emotions(frame)

    # Draw rectangles and labels
    for face in result:
        (x, y, w, h) = face['box']
        emotion, score = max(face['emotions'].items(), key=lambda x: x[1])
        color = (0, 255, 0) if emotion != 'neutral' else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()