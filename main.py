from ultralytics import YOLO
import cv2
import math

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Store previous center positions
previous_positions = {}

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # Detection + tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml"
    )

    # Get the first result
    result = results[0]

    # Draw detections
    annotated_frame = result.plot()

    # Check whether tracking IDs exist
    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = box

            # Calculate center of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_position = (center_x, center_y)

            # Check previous position
            if track_id in previous_positions:

                previous_position = previous_positions[track_id]

                # Calculate movement distance
                distance = math.sqrt(
                    (center_x - previous_position[0]) ** 2 +
                    (center_y - previous_position[1]) ** 2
                )

                # Determine movement status
                if distance > 5:
                    movement = "MOVING"
                else:
                    movement = "STATIONARY"

                # Display movement information
                cv2.putText(
                    annotated_frame,
                    f"ID {track_id}: {movement}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Update position
            previous_positions[track_id] = current_position

    # Display frame
    cv2.imshow(
        "AI Shoplifting Detection - Behaviour Analysis",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()