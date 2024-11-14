---
title: Neural networks
teaching: 20
exercises: 10
---

::::::::::::::::::::::::::::::::::::::: objectives

- Become familiar with key components of a neural network.
- Create the architecture for a convolutational neural network.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- What is a neural network?
- What are the characteristics of a dense layer?
- What is an activation function?
- What is a convolutional neural network?

::::::::::::::::::::::::::::::::::::::::::::::::::

## What is a neural network?

An artificial neural network, or just "neural network", is a broad term that describes a family of machine learning models that are (very!) loosely based on the neural circuits found in biology.

The smallest building block of a neural network is a single neuron. A typical neuron receives inputs (x1, x2, x3) which are multiplied by learnable weights (w1, w2, w3), then summed with a bias term (b). An activation function (f) determines the neuron output.

![](fig/neuron.png){alt='Neuron' width="600px"}

From a high level, a neural network is a system that takes input values in an "input layer", processes these values with a collection of functions in one or more "hidden layers", and then generates an output such as a prediction. The network has parameters that are systematically tweaked to allow pattern recognition.

![](fig/simple_neural_network.png){alt='Neuron' width="800px"}

The layers shown in the network above are "dense" or "fully connected". Each neuron is connected to all neurons in the preceeding layer. Dense layers are a common building block in neural network architectures.

"Deep learning" is an increasingly popular term used to describe certain types of neural network. When people talk about deep learning they are typically referring to more complex network designs, often with a large number of hidden layers.

## Activation Functions

Part of the concept of a neural network is that each neuron can either be 'active' or 'inactive'. This notion of activity and inactivity is attempted to be replicated by so called activation functions. The original activation function was the sigmoid function (related to its use in logistic regression). This would make each neuron's activation some number between 0 and 1, with the idea that 0 was 'inactive' and 1 was 'active'.

As time went on, different activation functions were used. For example the tanh function (hyperbolic tangent function), where the idea is a neuron can be active in both a positive capacity (close to 1), a negative capacity (close to -1) or can be inactive (close to 0).

The problem with both of these is that they suffered from a problem called [model saturation](https://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_SSCI_2015/data/7560b423.pdf). This is where very high or very low values are put into the activation function, where the gradient of the line is almost flat. This leads to very slow learning rates (it can take a long time to train models with these activation functions).

Another very popular activation function that tries to tackle this is the rectified linear unit (ReLU) function. This has 0 if the input is negative (inactive) and just gives back the input if it is positive (a measure of how active it is - the metaphor gets rather stretched here). This is much faster at training and gives very good performance, but still suffers model saturation on the negative side. Researchers have tried to get round this with functions like 'leaky' ReLU, where instead of returning 0, negative inputs are multiplied by a very small number.

![](fig/ActivationFunctions.png){alt='Activation functions' width="600px"}

## Convolutional neural networks

Convolutional neural networks (CNNs) are a type of neural network that especially popular for vision tasks such as image recognition. CNNs are very similar to ordinary neural networks, but they have characteristics that make them well suited to image processing.

Just like other neural networks, a CNN typically consists of an input layer, hidden layers and an output layer. The layers of "neurons" have learnable weights and biases, just like other networks.

What makes CNNs special? The name stems from the fact that the architecture includes one or more convolutional layers. These layers apply a mathematical operation called a "convolution" to extract features from arrays such as images.

In a convolutional layer, a matrix of values referred to as a "filter" or "kernel" slides across the input matrix (in our case, an image). As it slides, values are multiplied to generate a new set of values referred to as a "feature map" or "activation map".

![](fig/convolution_plotke.gif){alt='2D Convolution Animation by Michael Plotke' width="500px"}

Filters provide a mechanism for emphasising aspects of an input image. For example, a filter may emphasise object edges. See [setosa.io](https://setosa.io/ev/image-kernels/) for a visual demonstration of the effect of different filters.

## Creating a convolutional neural network

Before training a convolutional neural network, we will first need to define the architecture. We can do this using the Keras and Tensorflow libraries.

```python, python
# Create the architecture of our convolutional neural network, using
# the tensorflow library
from tensorflow.random import set_seed
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# set random seed for reproducibility
set_seed(42)
 
# Our input layer should match the input shape of our images.
# A CNN takes tensors of shape (image_height, image_width, color_channels)
# We ignore the batch size when describing the input layer
# Our input images are 256 by 256, plus a single colour channel.
inputs = Input(shape=(256, 256, 1))

# Let's add the first convolutional layer
x = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(inputs)

# MaxPool layers are similar to convolution layers. 
# The pooling operation involves sliding a two-dimensional filter over each channel of feature map and selecting the max values.
# We do this to reduce the dimensions of the feature maps, helping to limit the amount of computation done by the network.
x = MaxPool2D()(x)

# We will add more convolutional layers, followed by MaxPool
x = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(filters=12, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(filters=12, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(filters=20, kernel_size=5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(filters=20, kernel_size=5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(filters=50, kernel_size=5, padding='same', activation='relu')(x)

# Global max pooling reduces dimensions back to the input size
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
```

We can view the architecture of the model:

```python, python
model.summary()
```

```output
Model: "model_39"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_9 (InputLayer)        [(None, 256, 256, 1)]     0         
                                                                 
 conv2d_59 (Conv2D)          (None, 256, 256, 8)       80        
                                                                 
 max_pooling2d_50 (MaxPoolin  (None, 128, 128, 8)      0         
 g2D)                                                            
                                                                 
 conv2d_60 (Conv2D)          (None, 128, 128, 8)       584       
                                                                 
 max_pooling2d_51 (MaxPoolin  (None, 64, 64, 8)        0         
 g2D)                                                            
                                                                 
 conv2d_61 (Conv2D)          (None, 64, 64, 12)        876       
                                                                 
 max_pooling2d_52 (MaxPoolin  (None, 32, 32, 12)       0         
 g2D)                                                            
                                                                 
 conv2d_62 (Conv2D)          (None, 32, 32, 12)        1308      
                                                                 
 max_pooling2d_53 (MaxPoolin  (None, 16, 16, 12)       0         
 g2D)                                                            
                                                                 
 conv2d_63 (Conv2D)          (None, 16, 16, 20)        6020      
                                                                 
 max_pooling2d_54 (MaxPoolin  (None, 8, 8, 20)         0         
 g2D)                                                            
                                                                 
 conv2d_64 (Conv2D)          (None, 8, 8, 20)          10020     
                                                                 
 max_pooling2d_55 (MaxPoolin  (None, 4, 4, 20)         0         
 g2D)                                                            
                                                                 
 conv2d_65 (Conv2D)          (None, 4, 4, 50)          25050     
                                                                 
 global_average_pooling2d_8   (None, 50)               0         
 (GlobalAveragePooling2D)                                        
                                                                 
 dense_26 (Dense)            (None, 128)               6528      
                                                                 
 dropout_8 (Dropout)         (None, 128)               0         
                                                                 
 dense_27 (Dense)            (None, 32)                4128      
                                                                 
 dense_28 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 54,627
Trainable params: 54,627
Non-trainable params: 0
_________________________________________________________________
```



:::::::::::::::::::::::::::::::::::::::: keypoints

- Dense layers, also known as fully connected layers, are an important building block in most neural network architectures. In a dense layer, each neuron is connected to every neuron in the preceeding layer.
- Dropout is a method that helps to prevent overfitting by temporarily removing neurons from the network.
- The Rectified Linear Unit (ReLU) is an activation function that outputs an input if it is positive, and outputs zero if it is not.
- Convolutional neural networks are typically used for imaging tasks.

::::::::::::::::::::::::::::::::::::::::::::::::::


