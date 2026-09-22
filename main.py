from ultralytics import YOLO
import cv2
import math
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Previous positions of tracked persons
previous_positions = {}

# Store zone entry time for each person
zone_entry_times = {}

# Define shelf zone
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

            # Calculate center of person
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_position = (center_x, center_y)

            # -----------------------------
            # MOVEMENT ANALYSIS
            # -----------------------------

            movement = "STATIONARY"

            if track_id in previous_positions:

                previous_position = previous_positions[track_id]

                distance = math.sqrt(
                    (center_x - previous_position[0]) ** 2 +
                    (center_y - previous_position[1]) ** 2
                )

                if distance > 5:
                    movement = "MOVING"

            previous_positions[track_id] = current_position

            # -----------------------------
            # SHELF ZONE DETECTION
            # -----------------------------

            inside_zone = (
                shelf_x1 <= center_x <= shelf_x2
                and
                shelf_y1 <= center_y <= shelf_y2
            )

            # -----------------------------
            # ZONE TIMER
            # -----------------------------

            if inside_zone:

                # Start timer when person enters
                if track_id not in zone_entry_times:
                    zone_entry_times[track_id] = time.time()

                # Calculate time spent in zone
                duration = time.time() - zone_entry_times[track_id]

                zone_status = f"IN ZONE: {duration:.1f}s"

            else:

                # If person was previously in zone
                if track_id in zone_entry_times:

                    duration = time.time() - zone_entry_times[track_id]

                    print(
                        f"Person ID {track_id} "
                        f"spent {duration:.1f} seconds in shelf zone"
                    )

                    # Remove timer
                    del zone_entry_times[track_id]

                duration = 0
                zone_status = "OUTSIDE ZONE"

            # -----------------------------
            # DISPLAY INFORMATION
            # -----------------------------

            cv2.putText(
                annotated_frame,
                f"ID {track_id}",
                (int(x1), int(y1) - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                movement,
                (int(x1), int(y1) - 20),
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

    # Display frame
    cv2.imshow(
        "AI Shoplifting Detection - Behaviour Analysis",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()