# Localizing-human-pose-with-MPII-dataset



Experiments on adversarial learning for localizing the key-points in MPII Dataset


Pytorch implementation of chen et al. "Adversarial PoseNet" for landmark localization on digital images.
The architecture was  proposed by [Yu Chen, Chunhua Shen, Xiu-Shen Wei, Lingqiao Liu, Jian Yang](https://scholar.google.com/citations?user=IWZubqUAAAAJ&hl=zh-CN) in 
[Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389). 



## Lanmark localization 

Based on the given RGB dataset, the heatmap has been created with key-points. To prevent the model get over-fitting situation, data augmentation techniques,which is one of the regularization has been applied. 

Followings are the examples of the RGB images in the given dataset and it's heatmap.

<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/original.png" width="300px"/>
<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/original1.png" width="300px"/>

Followings are the examples of the RGB images in the given dataset and it's heatmap with key-points after applying the data augmentation techniques in data pre-processing part.

<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/augmented.png" width="300px"/>
<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/augmented1.png" width="300px"/>


##  Results Visualization
The results of this implementation:
In two different setup, the left side image is the ground-truth images with the key-points and the right side image is the output of the proposed model. In each images of the result,the red dot stands for the ground-truth key-points and the yellow dot stands for the prediction of the proposed model. 
For better understanding of the performance of the given model in two different setup,same image in the given dataset has been compared in two different settings.

### Adversarial PoseNet(In Adversarial setup using GAN framework):
<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/Screen%20Shot%202020-10-31%20at%202.57.09%20PM.png" width="200px"/><img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/adversarial.png" width="200px"/>


### Stack-hour-glass Network(In supervised setup):
<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/Screen%20Shot%202020-10-31%20at%202.57.09%20PM.png" width="200px"/><img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/supervised.png" width="200px"/>

### localization rate of diffent setups on the test split:

The PCKh@0.5 metrics has been used to measure the performance of the proposed model.
The best accuracy of the proposed model that have been trained in supervised way is 85.49%
The best accuracy of the proposed model that have been trained in adversarial way using GAN framework is 86.43%
As a result, the results shows that the proposed model perform better in adversarial way to predict the human key-points with MPII dataset.

<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/Screen%20Shot%202020-10-31%20at%202.56.16%20PM.png" width="300px"/>

<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/Screen%20Shot%202020-10-31%20at%202.55.31%20PM.png" width="300px"/>


## Main Prerequisites
- pytorch
- OpenCV
- Numpy
- Scipy-images
- ```The list of dependencies can be found in the the requirements.txt file. Simply use pip install -r requirements-pytorch-posenet.txt to install them.```


## Getting Started
### Installation
- Install Pytorch from https://pytorch.org/get-started/locally/
- Clone this repository:
```bash
git clone https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset.git
```


## Training and Test Details
To train a model, run any of the .sh file starting with "train". For example :  
```bash
train-ad-deafault-retrain.sh 
```
- A bash file has following configurations, that one can change 
```
python trainmodel-adversarial-mode-exp24.py \
--path mpii \
--modelName  trainmodel-adversarial-with-pretrain-defaultalpha-retrain \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--gpu_device 0 \
--lr .00025 \
--print_every 5000 \
--train_split 0.90 \
--loss mse \
--optimizer_type Adam \
--epochs 230 \
--dataset  'mpii' 
```
Models are saved to `./trainmodel/` (can be changed using the argument --modelName in .sh file).  

To test the model,
```bash
test-adversarial-with-pretrain-defaultalpha-retrain.sh
```

## Datasets


- ` MPII Dataset`: This dataset contains 25,000 images in 39 RGB nature. And this dataset covers 410 human activities that have been extracted from YouTube videos, and each image provided with an activity label.Each image contains 16 different co-ordinate body joints according to the human pose in every single image. Stored available co-ordinate joints in MPII dataset are right ankle, right knee, right hip, left hip, left knee, left ankle, pelvis, thorax, upper neck, head top, right wrist, right elbow, right shoulder, left shoulder, left elbow, left wrist. Every image has different size because it is not quadratic. But, most of the prominent person size is roughly 200 pixels length in the images.



<img src="https://github.com/YUNSUCHO/Localizing-human-pose-with-MPII-dataset/blob/main/READ/RGB.png" width="400px"/> 


## Reference
- The pytorch implementation of stacked-hour-glass, https://github.com/princeton-vl/pytorch_stacked_hourglass
- The pytorch implementation of self-adversarial pose estimation, https://github.com/roytseng-tw/adversarial-pose-pytorch
- The torch implementation of self-adversarial pose estimationh , https://github.com/dongzhuoyao/jessiechouuu-adversarial-pose


