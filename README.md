# ArmNN TFLite

### Install pip
```shell
sudo apt-get install python3-pip
```

### Install necessary python packages
```shell
pip3 install numpy==1.26.2 tflite_runtime pillow opencv librosa sounddevice
```

### Download ArmNN libraries
```shell
wget -O ArmNN-aarch64.tgz https://github.com/ARM-software/armnn/releases/download/v24.11/ArmNN-linux-aarch64.tar.gz
mkdir libs
tar -xvf ArmNN-aarch64.tgz -C libs
```

### Examples:

#### Simple Neural Networks
| model | description |
|---|---|
|[Sine Model](./sine_model/) | Basic Neural network TFLite model |
|[Digit recognize Model](./digit_recognize/) | Digit recognization model |

#### Audio Processing
| model | description |
|---|---|
| [Audio classifier Model](./audio_classifier/) | Audio classifier model |

#### Image processing
| model | description |
|---|---|
| [Mobilenet v1 Model](./mobilenet_v1/) | Mobilenet v1 image classification model |
| [YOLOv8n Model](./yolov8n/) | YOLOv8n image detection model |
| [SRGAN 4x Model](./srgan/) | SRGAN 4x image resolution upscale |
| [Auto crop Model](./auto_crop/) | Automatic document crop model |
