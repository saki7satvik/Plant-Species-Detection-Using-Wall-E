import pandas as pd
import numpy as np
from vosk import Model, KaldiRecognizer
import wave
import json
import pyaudio

model = Model("./vosk-model-small-en-us-0.15")

recognizer = KaldiRecognizer(model, 16000)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

print("Speak into the microphone...")

# df = pd.DataFrame({
#     "voice_data": [],
#     "actual_command": []
# })

try:
    while True:
        data = stream.read(4096, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            recognized_text = result['text']

            # Append the recognized text to the DataFrame
            # new_row = pd.DataFrame({"voice_data": [recognized_text], "actual_command": [""]})
            # df = pd.concat([df, new_row], ignore_index=True)

            print("Recognized text: ", recognized_text)

except KeyboardInterrupt:
    print("Exiting...")
    # df.to_csv('output_with_index.csv', index=True)
    print("Data saved to 'output_with_index.csv'")

finally:
    # Close the audio stream properly
    stream.stop_stream()
    stream.close()
    audio.terminate()      
