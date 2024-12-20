# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load YOLO model
# model = YOLO('plant_detection.pt')  # Use 'yolov5s.pt', 'yolov8n.pt' or your custom model

# # Initialize video capture (0 for webcam or provide video file path)
# cap = cv2.VideoCapture(0)  # Use 'video.mp4' for video file

# # Define class names (COCO classes)
# class_names = model.names

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Run inference
#     results = model(frame)

#     # results.xyxy[0] gives you the bounding boxes and related data
#     # results.pandas().xywh gives you the bounding boxes as pandas DataFrame (alternative)
#     for result in results:
#         boxes = result.boxes  # bounding boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
#             conf = box.conf[0]  # Confidence score
#             class_id = int(box.cls[0])  # Class ID
#             label = f"{class_names[class_id]} {conf:.2f}"

#             # Draw bounding boxes
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow("YOLO Object Detection", frame)

#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
import os
import cv2
import numpy as np
import pyttsx3  # Text-to-speech library
from ultralytics import YOLO

# Load YOLO model
model = YOLO('plant_detection.pt')  # Use your custom-trained model

# Initialize TTS engine
engine = pyttsx3.init()

# Set voice properties for a pleasant voice
voices = engine.getProperty('voices')
for voice in voices:
    if "Zira" in voice.name:  # Adjust to find the Zira voice
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 150)  # Adjust speech rate
engine.setProperty('volume', 1.0)  # Set maximum volume

# Define class names (update based on your model's classes)
class_names = model.names

# To avoid repeated speech
spoken_classes = set()

# Information functions
def rose_information():
    return (
        "This is Rose pant!.A rose plant is a woody perennial flowering plant of the genus Rosa, in the family Rosaceae. "
        "Roses are known for their stunning blooms and symbolic connection to love and romance."
    )

def cactus_information():
    return (
        "This is a Cactus!.A cactus is a member of the plant family Cactaceae, known for its thick stems and spines. "
        "They are survivalists, thriving in arid regions and surprising us with vibrant flowers."
    )

def speak_information(info):
    """Speak the given information using a pleasant voice."""
    engine.say(info)
    engine.runAndWait()

# Initialize video capture
cap = cv2.VideoCapture(1)  # Use webcam (0) or a video file path
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO inference
    results = model(frame)

    for result in results:
        boxes = result.boxes  # bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates of the bounding box
            conf = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = f"{class_names[class_id]} {conf:.2f}"

            # Draw bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Speak information based on class name
            class_name = class_names[class_id].lower()
            if class_name == "rose" and class_name not in spoken_classes:
                info = rose_information()
                print(info)
                speak_information(info)
                spoken_classes.add(class_name)
            elif class_name == "cactus" and class_name not in spoken_classes:
                info = cactus_information()
                print(info)
                speak_information(info)
                spoken_classes.add(class_name)

    # Show the frame
    cv2.imshow("YOLO Object Detection with Voice Feedback", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
