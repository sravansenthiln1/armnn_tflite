# Audio classification using Tensorflow Lite
This example uses a simple CNN model that recognizes a keyword from an audio file.
to showcase the working of TFLite delegate running with ArmNN acceleration.

Refer to the `audio_classification.ipynb` notebook on how to build this model yourself.

**Requirements for running the notebook:**

```
tensorflow==2.14.0
matplotlib==3.8.2
librosa==0.10.1
```

You will need to have the following file structure for the dataset.
```
├── audio_classifier_model.ipynb
└── datasets
    ├── edge
    ├── hello
    ├── khadas
    ├── mind
    ├── none
    ├── tone
    └── vim
```
Each subfolder should have 5 second .wav files named in order from 0 to 19,
such as edge-0.wav, edge-1.wav,... edge-19.wav

This model can identify the following keywords
0. none (no keyword said)
1. hello
2. khadas
3. vim
4. edge
5. tone
6. mind


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
