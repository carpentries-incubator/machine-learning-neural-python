---
title: "Neural networks"
teaching: 20
exercises: 10
questions:
- "What is a neural network?"
- "What is a convolutional neural network?"
objectives:
- "Create the architecture for a convolutational neural network."
keypoints:
- "Convolutional neural networks are typically used for imaging tasks."
---

## What is a neural network?

An artificial neural network, or just “neural network”, is a broad term that describes a family of machine learning models that are (very!) loosely based on the neural circuits found in biology.

The smallest building block of a neural network is a single neuron. A typical neuron receives inputs (x1, x2) which are multiplied by learnable weights (w1, w2), then summed with a bias term (b). An activation function determines the neuron output.

![Neuron](../fig/neuron.jpg){: width="600px"}

From a high level, a neural network is a system that takes input values in an “input layer”, processes these values with a collection of functions in one or more “hidden layers”, and then generates an output such as a prediction. The network has parameters that are systematically tweaked to allow pattern recognition.

![Neuron](../fig/simple_neural_network.jpg){: width="800px"}

“Deep learning” is an increasingly popular term used to describe neural networks. When people talk about deep learning they are typically referring to more complex network designs, often with a large number of hidden layers.

## Convolutional neural networks

Convolutional neural networks (CNNs) are a type of neural network that especially popular for vision tasks such as image recognition. CNNs are very similar to ordinary neural networks, but they have characteristics that make them well suited to image processing.

Just like other neural networks, a CNN typically consists of an input layer, hidden layers and an output layer. The layers of "neurons" have learnable weights and biases, just like other networks.

What makes CNNs special? The name stems from the fact that the architecture includes one or more convolutional layers. These layers apply a mathematical operation called a "convolution" to extract features from arrays such as images.

In a convolutional layer, a matrix of values referred to as a "filter" or "kernel" slides across the input matrix (such as an image). As it slides, the values are multiplied to generate a new set of values referred to as a "feature map" or "activation map".

![Example convolution operation](../fig/placeholder.png){: width="800px"}

Different filters allow different aspects of an input image to be emphasised. For example, certain filters may help to identify the edges of objects in an image.

## Creating a convolutional neural network

Before training a convolutional neural network, we will first need to define the architecture. We can do this using the Keras and Tensorflow libraries.

```python
# NOTE: CONSIDER REPLACING WITH
# https://www.tensorflow.org/tutorials/images/cnn


# In this step we will create the architecture of our convolutional neural network,
#We will use the Keras library, suitable for Deep Learning in Python
#Initially, we are going to import the Keras functions that we are going to use:

from keras import optimizers
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.models import Model
from keras.layers import Input, add
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
 
# Our input layer should match the input shape of our images.
# A CNN takes tensors of shape (image_height, image_width, color_channels)
# We ignore the batch size when describing the input layer
# Our input images are 256 by 256, plus a single colour channel.
inputs = Input(shape=(256, 256, 1))

# Let's add the first convolutional layer
x = Conv2D(8, 3, padding='same', activation='relu')(inputs)

# MaxPool layers are similar to convolution layers. The pooling operation involves sliding a two-dimensional filter over each channel of feature map and summarising the features.
# We do this to reduce the dimensions of the feature maps, helping to limit the amount of computation done by the network.
x = MaxPool2D()(x)

# We will add more convolutional layers, followed by MaxPool
x = Conv2D(8, 3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(12, 3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(12, 3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(20, 5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(20, 5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(50, 5, padding='same', activation='relu')(x)
x = GlobalAveragePooling2D()(x)

# Finally we will add two "dense" or "fully connected layers".
# Dense layers help with the classification task, after features are extracted.
x = Dense(128, activation='relu')(x)

# Dropout is a technique to help prevent overfitting that involves deleting neurons.
x = Dropout(0.6)(x)

x = Dense(32, activation='relu')(x)

# Our final dense layer has a single output to match the output classes.
# If we had multi-classes we would match this number to the number of classes.
outputs = Dense(1, activation='sigmoid')(x)

# Finally, we will define our network with the input and output of the network
model = Model(inputs=inputs, outputs=outputs)

# We define the network optimization method: ADAM, with the learning and decay rate
custom_adam = tf.optimizers.Adam(learning_rate=0.0005, decay=0.0002)
```
{: .language-python}

We can view the architecture of the model:

```python
model.summary()
```
{: .language-python}



{% include links.md %}
 



