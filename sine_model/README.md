# Basic Neural Network using Tensorflow Lite
This example uses a basic Neural Network that predicts the value of sin(x) 
to showcase the working of TFLite delegate running with ArmNN acceleration.

Refer to the `1_variable_regression_model.ipynb` notebook on how to build this model yourself.

**Requirements for running the notebook:**

```
tensorflow==2.15.0
matplotlib==3.8.2
```

### Add symlinks for libraries
```shell
sudo ln ../libs/libarmnnDelegate.so.29.0 libarmnnDelegate.so.29
sudo ln ../libs/libarmnn.so.33.1 libarmnn.so.33
```

### Run the example
```shell
python3 run_inference.py
```

### Note:
Modify the `BACKEND` variable in the code to use either `GpuAcc` or `CpuAcc` backends.
