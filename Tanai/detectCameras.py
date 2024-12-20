import cv2

def list_available_cameras():
    available_cameras = []
    max_tested = 10  # Number of camera indices to test

    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()

    if available_cameras:
        print("Available Camera Indices:")
        for idx in available_cameras:
            print(f"Camera Index: {idx}")
    else:
        print("No cameras found.")

if __name__ == "__main__":
    list_available_cameras()
