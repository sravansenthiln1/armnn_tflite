# SRGAN 4x image resolution upscale using Tensorflow Lite
This example uses [SRGAN](https://arxiv.org/abs/1609.04802)
to showcase the working of TFLite delegate running with ArmNN acceleration.

### Add symlinks for libraries
```shell
sudo ln ../libs/libarmnnDelegate.so.29.0 libarmnnDelegate.so.29
sudo ln ../libs/libarmnn.so.33.1 libarmnn.so.33
```

### Run the example
```shell
python3 run_inference.py
```

The upscaled output is saved as `output.png`

### Note:
Modify the `BACKEND` variable in the code to use either `GpuAcc` or `CpuAcc` backends.
