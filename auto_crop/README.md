# Document crop using encoder-decoder architecture with tensorflow lite
This example uses a simple Convolutional Encoder-Decoder that takes a downsampled image and segments the document,
and computes anchors to crop the document from the image to showcase the working of TFLite delegate running with ArmNN acceleration.

Refer to the `auto_crop_enc_dec_lite_model.ipynb` notebook on how to build this model yourself.

**Requirements for running the notebook:**

```
tensorflow==2.15
matplotlib==3.8.2
opencv-python==4.5.5
```

The dataset should be a folder of the following structure
```
└── dataset
    ├── anchors
    │   ├── 0.json
    │   ├── 1.json
    │    ...
    │   └── 169.json
    └── images
        ├── 0.png
        ├── 1.png
        ...
        └── 0169.jpg
```

The dataset was made using the images provided by [WarpDoc perspective dataset](https://sg-vilab.github.io/event/warpdoc/)

Annotations made using labelme tool.

### Add symlinks for libraries
```shell
sudo ln ../libs/delegate/libarmnnDelegate.so.29.1 libarmnnDelegate.so.29
sudo ln ../libs/libarmnn.so.34.0 libarmnn.so.34
```

### Run the example
```shell
python3 run_inference.py
```

The output is in `output.png`

### Note:
Modify the `BACKEND` variable in the code to use either `GpuAcc` or `CpuAcc` backends.
