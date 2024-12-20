import cv2
import numpy as np
import os
import threading
import speech_recognition as sr

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Directly point to file

# Initialize OpenCV LBPH Face Recognizer
# For OpenCV 4.x+ versions, we should use the 'opencv-contrib-python' package
face_recognizer = cv2.face_LBPHFaceRecognizer.create()  # Correct method for recent OpenCV

# Directory to store saved face images
FACE_DATA_DIR = "faces"
os.makedirs(FACE_DATA_DIR, exist_ok=True)

person_names = {}  # Map labels to person names
face_counter = 1  # Label for new faces
lock = threading.Lock()  # Synchronize threads
active_listening = False  # Flag to prevent multiple threads
unknown_face_counter = 0  # Counter for stable detection
MAX_STABLE_FRAMES = 15  # Number of frames to consider a face stable

def save_face_snapshots(face, label):
    """Save multiple snapshots of a face with variations."""
    
    # Define the number of rotations you want (angles)
    angles = [0, 10, 20, 30, 40, -10, -20, -30]
    
    # Define brightness and contrast variations
    brightness_contrast_variations = [
        (1.5, 50),   # Bright image
        (0.7, -30),  # Dark image
        (1.2, 10),   # Slightly brighter
        (1.0, 0),    # Normal brightness
    ]
    
    # Define blur kernel sizes
    blur_variations = [
        (5, 5),  # Slight blur
        (9, 9),  # More blur
    ]
    
    # Save original image first
    cv2.imwrite(os.path.join(FACE_DATA_DIR, f"face_{label}_original.jpg"), face)
    
    # Rotation variations
    for angle in angles:
        rotated_face = rotate_image(face, angle)
        cv2.imwrite(os.path.join(FACE_DATA_DIR, f"face_{label}_rotated_{angle}.jpg"), rotated_face)
    
    # Brightness and contrast variations
    for alpha, beta in brightness_contrast_variations:
        bright_face = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)
        cv2.imwrite(os.path.join(FACE_DATA_DIR, f"face_{label}_bright_contrast_{alpha}_{beta}.jpg"), bright_face)
    
    # Blur variations
    for kernel_size in blur_variations:
        blurred_face = cv2.GaussianBlur(face, kernel_size, 0)
        cv2.imwrite(os.path.join(FACE_DATA_DIR, f"face_{label}_blur_{kernel_size[0]}x{kernel_size[1]}.jpg"), blurred_face)

def rotate_image(image, angle):
    """Rotate the image by a specific angle."""
    # Get the image center
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # Rotate the image around its center
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def load_faces_and_train():
    """Load saved face images and retrain recognizer."""
    images, labels = [], []
    for file in os.listdir(FACE_DATA_DIR):
        if file.startswith("face_") and file.endswith(".jpg"):
            try:
                label = int(file.split("_")[1])
                img_path = os.path.join(FACE_DATA_DIR, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)
            except (IndexError, ValueError):
                print(f"Skipping invalid file: {file}")
    if images:
        face_recognizer.train(images, np.array(labels))

def voice_input_handler(face_roi):
    """Listen for voice commands and save face with the given name."""
    global face_counter, active_listening
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for voice commands...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"Voice Command: {command}")

            if "my name is" in command:
                # Extract the person's name
                name = command.replace("my name is", "").strip().title()
                with lock:
                    person_names[face_counter] = name
                    save_face_snapshots(face_roi, face_counter)
                    load_faces_and_train()
                    print(f"Face registered as {name}!")
                    face_counter += 1
        except sr.WaitTimeoutError:
            print("Voice recognition timed out.")
        except Exception as e:
            print(f"Voice recognition error: {e}")
    active_listening = False  # Reset the flag when done

def is_face_known(face_roi):
    """Check if the face is already known."""
    try:
        label, confidence = face_recognizer.predict(face_roi)
        if confidence < 80:  # Confidence threshold for matching
            return label, confidence
    except:
        pass
    return None, None

def process_camera_feed():
    global active_listening, unknown_face_counter
    cap = cv2.VideoCapture(0)
    load_faces_and_train()  # Load pre-existing faces

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (150, 150))  # Normalize face size

            label, confidence = is_face_known(face_resized)

            if label is None:  # Unknown face detected
                unknown_face_counter += 1  # Increment the counter
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red box

                # Start listening only after the face persists for MAX_STABLE_FRAMES
                if unknown_face_counter >= MAX_STABLE_FRAMES and not active_listening:
                    active_listening = True  # Prevent multiple threads
                    unknown_face_counter = 0  # Reset counter
                    voice_thread = threading.Thread(target=voice_input_handler, args=(face_resized,))
                    voice_thread.start()
            else:
                # Known face: reset the unknown face counter
                unknown_face_counter = 0
                name = person_names.get(label, f"Person {label}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Detection and Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Press 'q' to quit.")
    process_camera_feed()
