# Image classification using Tensorflow Lite
This example uses the [mobilenet_v1 model](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) 
to showcase the working of TFLite delegate running with ArmNN acceleration.

## Add symlinks for libraries
```shell
sudo ln ../libs/libarmnnDelegate.so.29 libarmnnDelegate.so.29
sudo ln ../libs/libarmnn.so.33 libarmnn.so.33
```

### Run the example
```shell
python3 run_inference.py
```

### Note:
Modify the `BACKEND` variable in the code to use either `GpuAcc` or `CpuAcc` backends.
