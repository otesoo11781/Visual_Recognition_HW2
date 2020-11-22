# Visual_Recognition_HW2

This is howework 2 for selected topics in visual recongnition using deep learning. The goal is to detect the house number.

I use Yolov4, which is a SOTA and most popular object detector, to detect the digits for Street View House Number (SVHN) dataset. 

Yolov4 is pretrained on MS COCO dataset, and then fine tuned on SVHN training dataset.

When run on Colab, it can achieve **0.49 mAP** and about **25 fps** on SVHN testing dataset.

Yolov4 uitilize a large amount of novel model tricks to improve the network, the details please refer to the [original paper](https://arxiv.org/abs/2004.10934).

The source code is highly borrowed from [Yolov4](https://github.com/AlexeyAB/darknet) and [SVHN](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN).

## Hardware
The following specs were used to **train** the yolov4 on SVHN:
- Ubuntu 16.04 LTS
- 2x RTX 2080 with CUDA=10.1

The following specs were used to **test** the yolov4 on SVHN:
- Tesla T4 with CUDA=10.1 on Colab


## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Transfer Training](#transfer-training)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation
Using Anaconda is strongly recommended.
```
conda create -n [env_name] python=3.7
conda activate [env_name]
pip install opencv-python
conda install pandas
conda install h5py
```

Then, install the Yolov4 by following command:
```
cd darknet
make
```

If there is any problem when install Yolov4, please refer to the [Yolov4](https://github.com/AlexeyAB/darknet).

## Dataset Preparation
Download the dataset from the [Google drive](https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl) which is provided by TAs

Then, unzip them and put it under the **./darknet/DigitDetection/dataset/** directory

Hence, the data directory is structured as:
```
./darknet/
  +- DigitDetection/
  |  +- dataset/
     |  +- train/
          |  +- 1.jpg ...
          |  +- digitStruct.mat
     |  +- test/
          |  +- 1.jpg ...
     |  construct_datasets.py    
```

Next, process the annotation file (digitStruct.mat) into yolov4 format by running:
```
cd DigitDetection/dataset/
python construct_datasets.py
cd ../../
```
After that, you will get a txt file, which contains the corresponding bounding boxes, for each training image in train/ directory.

**Important: you can download the processed dataset from [dataset.zip](https://drive.google.com/file/d/1dlNmVJmfG9Df9z21hZwKe9hR_h-dPhuG/view?usp=sharing)**

## Transfer Training
**Important: Download the required pretrained weights and transfer trained weights by [weights.zip](https://drive.google.com/file/d/16GZVXv3TJ7jCptoKbXecIS2hxq25dr3H/view?usp=sharing)**

- **yolov4.conv.137**: pretrained on MS COCO dataset
- **yolov4-HN_best.weights**: best weights trained on SVHN dataset in 20,000 iterations.
- **yolov4-HN_final.weights**: fianl weights trained on SVHN dataset in 20,000 iterations. 

Move all the weights to the **./darknet/DigitDetection/weights/** directory.
Hence, the weights directory is structured as:
```
./darknet/
  +- DigitDetection/
  |  +- weights/
     |  +- yolov4.conv.137
     |  +- yolov4-HN_best.weights
     |  +- yolov4-HN_final.weights
```

### Retrain the yolov4 model which is pretrained on MS COCO dataset (optional)
If you don't want to spend 4 days training a model, you can skip this step and just use the **yolov4-HN_best.weights** I provided to inference. 

Now, let's transfer train the yolov4.conv.137 on SVHN dataset:

1. please ensure there is MS COCO pretrained Yolov4 model (i.e. yolov4.conv.137).

2. modify ./DigitDetection/cfg/yolov4-HN.cfg file to training mode:
```
# Training
batch=64
subdivisions=64
```

3. run the following command:
```
./darknet detector train ./DigitDetection/cfg/HN.data ./DigitDetection/cfg/yolov4-HN.cfg ./DigitDetection/weights/yolov4.conv.137 -map -gpus 0,1
```
It takes about 3~4 days to train the model on 2 RTX 2080 GPUs.

Finally, we can find the best weights **yolov4-HN_best.weights** in **./darknet/DigitDetection/weights/** directory.


## Inference
With the testing dataset and trained model, you can run the following commands to obtain the prediction results:

1. modify ./DigitDetection/cfg/yolov4-HN.cfg file to testing mode:
```
# Testing
batch=1
subdivisions=1
```
2. run yolov4 detector:
```
./darknet detector test ./DigitDetection/cfg/HN.data ./DigitDetection/cfg/yolov4-HN.cfg ./DigitDetection/weights/yolov4-HN_best.weights -thresh 0.005 -dont_show -out ./DigitDetection/result.json < ./DigitDetection/cfg/test.txt
```

After that, you will get detection results (**./DigitDetection/result.json**).

Note: You can test my model on [Colab notebook](https://colab.research.google.com/drive/1cdcXTFOS86gu9-ziz4vtU19kIUxt_AtG?usp=sharing). It will show a inference time of **24.538 fps**.
**Note**: The repo has provided **result.json** which is inferred on Colab.

## Make Submission
1. Transform result.json into submission format by:
```
python ./DigitDetection/parse_result.py --input ./DigitDetection/result.json --output ./DigitDetection/0856610.json
```
2. Submit transformed **0856610.json** to [here](https://drive.google.com/drive/folders/1QNW9YvzFM7Nmg0PqUqbjgqpFyoo1wBEu).

**Note**: the repo has provided **0856610.json** which is corresponding to my submission with **mAP 0.49137**. 

