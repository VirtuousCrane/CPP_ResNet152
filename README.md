# CPP_ResNet152
## The Architecture Architecture
This Neural Network was built based on the Residual Network
architecture proposed in "Deep Residaul Learning for Image
Recognition" (He et al.). To be more specific, I have used
the bottleneck architecture proposed in this paper to
construct ResNet152. As for the dataset used to train this
network, I have decided to use some images from the famous 
2015 Imagenet Large Scale Visual Recognition Challenge
dataset (ILSVRC 2015) for training. I initially wanted to
train my Neural Network on all the images in this dataset
however, due to the time constraint and the specs of my
device, I have decided to only train on 15000 images
instead of the whole dataset, which consists of 50000
images.

## Basic Theory of Residual Networks
The Residual Network was introduced to address the degradation
problem of Neural Networks.

The main theory of this architecture is based on this formula:\
	F(x) := H(x) + x\
    WHERE\
	F(x) = Residual Function\
	H(x) = Desired Mapping\
