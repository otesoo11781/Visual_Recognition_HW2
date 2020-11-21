# Visual_Recognition_HW2

This is howework 2 for selected topics in visual recongnition using deep learning. The goal is to detect the house number.

I use Yolov4, which is a SOTA and most popular object detector, to detect the digits for Street View House Number (SVHN) dataset. 

Yolov4 is pretrained on MS COCO dataset, and then fine tuned on SVHN training dataset.

When run on Colab, it can achieve **0.49 mAP** and about **25 fps** on SVHN testing dataset.

Yolov4 uitilize a large amount of novel model tricks to improve the network, the details please refer to the [original paper](https://arxiv.org/abs/2004.10934).

The source code is highly borrowed from [Yolov4](https://github.com/AlexeyAB/darknet) and [SVHN](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN)

## Hardware
The following specs were used to **train** the yolov4 on SVHN:
- Ubuntu 16.04 LTS
- RTX 2080 with CUDA=10.1

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
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n [env_name] python=3.6
conda activate [env_name]
pip install -r requirements.txt
```

If error occurs during installation, please manually install them (refer to: [Pytorch](https://pytorch.org/get-started/locally/) and [Inplace_abn](https://github.com/mapillary/inplace_abn))

Note: Inplace_abn only supports Linux and CUDA version>=10.1

## Dataset Preparation
Download the dataset from the [kaggle website](https://www.kaggle.com/c/cs-t0828-2020-hw1/data)

Then, unzip them and put it under the **./data/** directory

Hence, the data directory is structured as:
```
./data
  +- training_data
  |  +- training_data
     |  +- 00001.jpg ...
  +- testing_data
  |  +- testing_data
     |  +- 00001.jpg ...
  +- training_labels.csv
```

## Transfer Training
### Retrain the model which pretrained on ImageNet
If you don't want to retrain the model, you can skip this step and download the trained weights on [Pretrained models](#pretrained-models)

Now, let's transfer train the model:

1. you should download the ImageNet pretrained TResNet model (e.g. tresnet_xl_448.pth) from [TResNet Model Zoo](https://github.com/mrT23/TResNet/blob/master/MODEL_ZOO.md).

2. put the model in the **./checkpoints/** directory.

3. run the following command:
```
$ python transfer.py --dataset_path=./data/training_data/training_data --label_path=./data/training_labels.csv --model_path=./checkpoints/tresnet_xl_448.pth --model_name=tresnet_xl --epochs=100 --batch_size=12
```
It takes about 17 hours to train the model and outputs 2 files:
1. **./checkpoints/transfer_model.pth**: the trained weights on given car brand dataset
2. **./class_name.npy**: the class names of car brand dataset

### Pretrained models
You can download pretrained model (**transfer_model.pth**) that used for my submission from [Here](https://drive.google.com/drive/folders/1Hj7sXE6OJt12IlDH7sQKuFU1l5dBZl1-?usp=sharing).

Then, put it under the **./checkpoints/** directory:
```
./checkpoints
  +- tresnet_xl_448.pth
  +- transfer_model.pth
```

In addition, the **class_name.npy** has been contained in this repo.

## Inference
With the testing dataset and trained model, you can run the following command to obtain the prediction results:
```
$ python test.py --test_dir=./data/testing_data/testing_data --model_path=./checkpoints/transfer_model.pth --model_name=tresnet_xl
```
After that, you will get classification results (**./result.csv**) which statisfy the submission format.

Note: please ensure **./class_name.npy** exists.

## Make Submission
Please go to [submission page](https://www.kaggle.com/c/cs-t0828-2020-hw1/submit) of kaggle website and upload the **result.csv** obtained in the previous step.

Note: the repo has provided **./result.csv** which is corresponding to my submission on leaderboard with accuracy 0.9304

