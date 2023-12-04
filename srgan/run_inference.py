#
# SRGAN 4x image resolution upscale accelerated with ArmNN
#

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

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
BACKENDS = 'CpuAcc'

# Set path to the TFLite experimental delegate Libraries
#
# Delegate path:
DELEGATE_PATH = "./libarmnnDelegate.so.29"

# Set path to the TFLite model
#
# Model path:
MODEL_PATH = "./srgan.tflite"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./sample.png"

img = Image.open(IMAGE_PATH)

armnn_delegate = tflite.load_delegate(
    library = DELEGATE_PATH,
    options = {
        "backends":BACKENDS,
        "logging-severity": "info"
    }
)

interpreter = tflite.Interpreter(
    model_path = MODEL_PATH,
    experimental_delegates = [armnn_delegate]
)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_details[0]['shape'][1] = img.height
input_details[0]['shape'][2] = img.width

interpreter.resize_tensor_input(0, input_details[0]['shape'])
interpreter.allocate_tensors()

img = np.expand_dims(img, 0)
img = (np.float32(img))

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

result = np.squeeze(output)
result_img = Image.fromarray(np.uint8(result))
result_img.save('output.png')
