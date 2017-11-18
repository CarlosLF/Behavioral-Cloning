# Behavioral-Cloning

## Introduction

In this project we use deep learning techniques to teach a car to drive autonoumously in a simulator. In a simulator the user can drive the car to collect training information. The simulator has also an autonomous mode were we can test the trained neural network.

## Network model

The model is based on the NVIDIA model arquitecture, with the adition of a dropout to decrease overfittiong.

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 33, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0

## Training dataset

The training dataset was obtained using the simulator. Then it was augmented using a generator.

The dataset was aumented using the following procedures

0. No image change

1. Image flip, the image is inverted and the angle sign is changed, with this image we can get a simulation of a car driving in the opossite direction.

2. Brightness change. The brightness of the image is changed.

3. Translate image, the image is shiftted n-random pixels, and the angle is changed proportionally to such displacement.

The generator uses the previous process to change the images for each image in the batch. The algorithm uses the left, right and center camera images.

## Training

The training was performed with a GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1

The network was trainned with augmented data with 20000 samples per epoch, with 3 epochs.

To train the network we use the following command

**python model.py**

20000/20000 [==============================] - 2036s 102ms/step - loss: 0.0355 -
 val_loss: 0.0282
Epoch 2/3
20000/20000 [==============================] - 1970s 98ms/step - loss: 0.0300 -
val_loss: 0.0220
Epoch 3/3
20000/20000 [==============================] - 1965s 98ms/step - loss: 0.0258 -
val_loss: 0.0159

## Testing

To test the network we have to run the simulator in autonomous mode, then we have to run the following command

**python drive.py**

The driving results are shown in the file video.mp4

