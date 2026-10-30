
from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Open laptop webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started! Press 'q' to quit.")

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform object detection
    results = model.predict(
        source=frame,
        conf=0.40,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")
