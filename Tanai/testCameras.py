import cv2

# Open the camera (0 is usually the default camera, use 1 or 2 for other connected cameras)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Start capturing and displaying video frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame is read correctly, ret will be True
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame in a window
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
