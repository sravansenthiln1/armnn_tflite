# Digit recognization with a simple CNN using tensorflow lite
This example uses a simple CNN model that takes a 28x28 pixel image and predicts what digit it is,
to showcase the working of TFLite delegate running with ArmNN acceleration.

Refer to the `digit_recognization_CNN.ipynb` notebook on how to build this model yourself.

**Requirements for running the notebook:**

```
tensorflow==2.15.0
matplotlib==3.8.2
```

### Add symlinks for libraries
```shell
sudo ln ../libs/delegate/libarmnnDelegate.so.29.1 libarmnnDelegate.so.29
sudo ln ../libs/libarmnn.so.34.0 libarmnn.so.34
```

### Run the example
```shell
python3 run_inference.py
```

### Note:
Modify the `BACKEND` variable in the code to use either `GpuAcc` or `CpuAcc` backends.
