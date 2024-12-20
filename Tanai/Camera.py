import cv2
from ultralytics import YOLO

class Eye:
    """Class to represent a camera (eye)."""
    def __init__(self, camera_index, name="Eye"):
        self.name = name
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open {self.name} camera with index {camera_index}")
    
    def read_frame(self, eye):
        ret, frame = self.cap.read()
        if eye == "right":
            return cv2.rotate(frame, cv2.ROTATE_180)
        if not ret:
            raise RuntimeError(f"Failed to read frame from {self.name}")
        return frame
    
    def release(self):
        self.cap.release()

def add_label(frame, label):
    """Add a label (text) to the top-left corner of the frame."""
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame

def main():
    # Initialize cameras (left and right eyes)
    left_eye = Eye(camera_index=0, name="Left Eye")
    right_eye = Eye(camera_index=1, name="Right Eye")
    
    # Load YOLOv8 model (pre-trained on COCO dataset)
    model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt', 'yolov8m.pt', etc. for larger models
    
    while True:
        # Capture frames
        left_frame = left_eye.read_frame("left")
        right_frame = right_eye.read_frame("right")

        # Add labels to identify which eye's feed it is
        left_frame = add_label(left_frame, "Left Eye")
        right_frame = add_label(right_frame, "Right Eye")

        # Run YOLOv8 detection on both frames with minimum confidence of 95%
        left_results = model(left_frame, conf=0.75, verbose=False)
        right_results = model(right_frame, conf=0.75, verbose=False)

        # Visualize detection results
        left_frame_detected = left_results[0].plot()
        right_frame_detected = right_results[0].plot()

        # Combine frames for stereo display
        combined_frame = cv2.hconcat([left_frame_detected, right_frame_detected])

        # Display the combined frame
        cv2.imshow("YOLOv8 Object Detection - Stereo Vision", combined_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    left_eye.release()
    right_eye.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
