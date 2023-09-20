# ArmNN TFLite

### Get the necessary system packages
```shell
sudo apt-get install git wget unzip zip python3 python3-pip
```

### Install the TFLite runtime interpreter
```shell
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### Download ArmNN libraries
```shell
wget -O ArmNN-aarch64.tgz https://github.com/ARM-software/armnn/releases/download/v23.08/ArmNN-linux-aarch64.tar.gz
mkdir libs
tar -xvf ArmNN-aarch64.tgz -C libs
```

Try the examples:

* [Sine Model](./sine_model/) - Basic Neural network TFLite model

* [Mobilenet v1 Model](./mobilenet_v1/) - Mobilenet v1 image classification model

