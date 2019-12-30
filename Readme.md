# Artifact detection for RADCURE

Convolutional neural net to detect artifact detection in the Radcure dataset images.

## Preprocessing
- Padding images to 512x512x512 and downsampling to 256x256x256 by default.

## Augmentation
- Horizontal mirroring of images

## Sampling
- Weighted Random Sampler with weights equal to the inverse of the class size

## Neural Net
- Standard 3d Convolution layers with Batch Normalizatin and Maxpooling except for the las tlayer which average pools
- Fully connected layers with sigmoid output
- Using Leaky-Relu as the activation function
- Binary Cross Entropy as loss function
- using stochastic gradient descent with a step scheduler

Results can be found [Here](./output.org "Results") 

## Requirements:
- Python
- Pytorch
- Numpy
- Sklearn
- Skimage
