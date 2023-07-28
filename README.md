# RiverSemanticSegmentation

## Overview

The repository contains a code that allows training models based on convolutional neural networks for segmenting river areas in satellite images composed of RGB visible bands.

## Results
The author's implementation of the vgg_unet model scored IoU=0.90174. Below is a sample data (columns: input, model output, model output, respectively).

![results.png](https://i.postimg.cc/Hk06sPNr/results.png)

## Tools used
- PyTorch - ML framework
- OpenCV - a library for image processing
- NumPy - a library for matrix operations
- neptune - logging tool

## Dataset
Dataset available for download from a separate repository: https://github.com/shocik/sentinel-river-segmentation-dataset

## Running the solution
Running the code on your own computer requires the following preparatory steps:

1. Neptune configuration in file [config.cfg](config.cfg).
2. Modify the path to the working directory in the file [train.ipynb](train.ipynb):
    ```Python
    #set workdir
    os.chdir("/content/drive/MyDrive/RiverSemanticSegmentation/")
    ```
3. Modifying the path to a dataset in a file [train.ipynb](train.ipynb):
    ```Python
    #dataset configuration
    dataset_dir = os.path.normpath("/content/drive/MyDrive/SemanticSegmentationV2/dataset/")
    ```
