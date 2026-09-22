from ultralytics import YOLO
import cv2
import math

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Previous positions of tracked persons
previous_positions = {}

# Define shelf zone
# Format: x1, y1, x2, y2
shelf_x1 = 150
shelf_y1 = 100
shelf_x2 = 500
shelf_y2 = 400

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # YOLO detection + tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml"
    )

    result = results[0]

    # Draw YOLO detections
    annotated_frame = result.plot()

    # Draw shelf zone
    cv2.rectangle(
        annotated_frame,
        (shelf_x1, shelf_y1),
        (shelf_x2, shelf_y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        "SHELF ZONE",
        (shelf_x1, shelf_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # Check tracking IDs
    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = box

            # Calculate person center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_position = (center_x, center_y)

            # Check whether person is inside shelf zone
            inside_zone = (
                shelf_x1 <= center_x <= shelf_x2
                and
                shelf_y1 <= center_y <= shelf_y2
            )

            # Calculate movement
            movement = "STATIONARY"

            if track_id in previous_positions:

                previous_position = previous_positions[track_id]

                distance = math.sqrt(
                    (center_x - previous_position[0]) ** 2 +
                    (center_y - previous_position[1]) ** 2
                )

                if distance > 5:
                    movement = "MOVING"

            # Display status
            if inside_zone:
                zone_status = "IN SHELF ZONE"
            else:
                zone_status = "OUTSIDE ZONE"

            cv2.putText(
                annotated_frame,
                f"ID {track_id}",
                (int(x1), int(y1) - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                movement,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                zone_status,
                (int(x1), int(y2) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Update position
            previous_positions[track_id] = current_position

    # Display
    cv2.imshow(
        "AI Shoplifting Detection - Zone Analysis",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()