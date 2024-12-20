import threading
import cv2
import os
import serial
import numpy as np
import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import time
from ultralytics import YOLO

# Arduino Serial Communication Setup
class ArduinoController:
    def __init__(self, port='COM3', baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None

    def connect(self):
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Connected to Arduino on port {self.port}")
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")

    def send_command(self, command):
        if self.connection:
            try:
                self.connection.write((command + '\n').encode())
                print(f"Command sent: {command}")
            except Exception as e:
                print(f"Error sending command: {e}")
        else:
            print("Arduino not connected.")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Arduino connection closed.")

# Machine Learning Model for Command Prediction
class CommandPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def load_and_train(self):
        df = pd.read_csv(self.data_path)
        df['voice_data'] = df['voice_data'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        X = self.vectorizer.fit_transform(df['voice_data'])
        y = df['actual_command']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        print("Command prediction model trained.")

    def predict(self, text):
        processed_text = re.sub(r'[^\w\s]', '', text.lower())
        vectorized_text = self.vectorizer.transform([processed_text])
        return self.model.predict(vectorized_text)[0]

# YOLO Detection
class YOLODetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def start_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Starting YOLO detection...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            results = self.model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{self.class_names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('YOLO Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Voice Recognition
class VoiceRecognition:
    def __init__(self, predictor, controller):
        self.predictor = predictor
        self.controller = controller

    def start_recognition(self):
        model = Model("vosk-model-small-en-us-0.15")
        recognizer = KaldiRecognizer(model, 16000)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
        stream.start_stream()

        print("Voice recognition started. Speak into the microphone...")
        try:
            while True:
                data = stream.read(4096, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    command = result.get("text", "")
                    print(f"Recognized command: {command}")

                    action = self.predictor.predict(command)
                    print(f"Predicted action: {action}")
                    self.controller.send_command(action)
        except KeyboardInterrupt:
            print("Voice recognition stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

# Main Program
if __name__ == "__main__":
    arduino_controller = ArduinoController(port="COM6")
    arduino_controller.connect()

    command_predictor = CommandPredictor(data_path="voice_recognition_dataset.csv")
    command_predictor.load_and_train()

    yolo_detection = YOLODetection(model_path="plant_detection.pt")
    voice_recognition = VoiceRecognition(predictor=command_predictor, controller=arduino_controller)

    camera_thread = threading.Thread(target=yolo_detection.start_detection, daemon=True)
    voice_thread = threading.Thread(target=voice_recognition.start_recognition, daemon=True)

    camera_thread.start()
    voice_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Shutting down Wall-E...")
        arduino_controller.disconnect()
