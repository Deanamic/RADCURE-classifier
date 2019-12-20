# Artifact detection for RADCURE

Convolutional neural net to detect artifact detection in the Radcure dataset images.

## Preprocessing
- Padding images to 512x512x512 and downsamples to 256x256x256 by default.

## Augmentation
- Horizontal mirroring of images

## Sampling
- Weighted Random Sampler with weights equal to the inverse of the class size

## Neural Net
6 layers of Conv3d with:
- Dropout (p = 0.2)
- BatchNorm
- LeakyReLU as activation function

And 3 fully connected layers, with a sigmoid at the output

Loss function is Binary Cross Entropy

## Requirements:
- Python
- Pytorch
- Numpy
- Sklearn
- Skimage
