#
# Audio classification inference accelerated with ArmNN
#

import numpy as np
import tflite_runtime.interpreter as tflite
import sounddevice as sd
import librosa
import sys

# ArmNN supports running TFLite models on both Arm Cortex-A CPUs and Arm Mali GPUs
#
# GpuAcc: GPU accelerated backend, Uses the GPU compute units for model inferencing
#     -> note: there is a generally longer initialization delay on the GPU but execution times will be shorter
#
# CpuAcc: Cpu Accelerated backend, Uses CPU optimized model inferencing
#     -> note: there is a shorter initialization delay on the CPU but execution times can vary
#
# CpuRef: Reference backend for running on the CPU, extremely slow
#
# Set Backend variable based on preference priority
#
# Preferred Backends: "GpuAcc,CpuAcc,CpuRef"
BACKENDS = "GpuAcc"

# Set path to the TFLite experimental delegate Libraries
#
# Delegate path:
DELEGATE_PATH = "./libarmnnDelegate.so.29"

# Set path to the TFLite model
#
# Model path:
MODEL_PATH = "./audio_classifier.tflite"

# Set path to the input audio (for this example)
#
# Audio path:
AUDIO_PATH = "./sample.wav"

# Other audio parameters
DURATION = 5
SR = 22050

# Map the tag output to the appropriate string
TAGS = {
        0:'none',
        1:'hello',
        2:'khadas',
        3:'vim',
        4:'edge',
        5:'tone',
        6:'mind'
}

def audio_callback(indata, frames, time, status):
    if status:
        print(f"error: {status}")
    audio_data.append(indata.copy())

try:
    if (sys.argv[1] == 'm'):
        audio_data = []
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SR):
            print(f"Recording {DURATION} seconds of audio...")
            sd.sleep(int(DURATION * 1000))
            print("Recording complete.")

        scale = np.concatenate(audio_data, axis=0)
        scale = scale[:110250, 0]
        sr = SR
except:
    scale, sr = librosa.load(AUDIO_PATH)

mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=4096, hop_length=512, n_mels=256, fmax=8000)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)

min_db = np.min(log_mel_spectrogram)
max_db = np.max(log_mel_spectrogram)

log_mel_spectrogram = (log_mel_spectrogram - min_db) / (max_db - min_db)

armnn_delegate = tflite.load_delegate(
    library = DELEGATE_PATH,
    options = {
        "backends":BACKENDS,
        "logging-severity": "info",
    }
)

interpreter = tflite.Interpreter(
    model_path = MODEL_PATH,
    experimental_delegates = [armnn_delegate]
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], [log_mel_spectrogram])
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])

prediction = np.argmax(output_data[0])

print(prediction, TAGS[prediction])
