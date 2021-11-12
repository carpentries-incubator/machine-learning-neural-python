---
title: "Applied machine learning"
teaching: 20
exercises: 10
questions:
- "How do we apply neural networks to a classification task?"
objectives:
- "Classify pneumonia in chest X-rays."
keypoints:
- "Neural networks can help with real world tasks."
---

## Learning to recognise disease in X-rays

Pneumonia is an infection of one or both of the lungs. When someone has pneumonia, the lungs are inflamed and may be partially filled with fluid. Symptoms often include difficulty breathing, chest pain, cough, and fatigue.

A chest radiograph - or chest X-ray - is an image of the chest used to diagnose conditions affecting the heart and lungs such as pneumonia. One of the signs of pneumonia is an increased whiteness in the lungs as a result of increased density.

The following chest X-ray shows signs of pneumonia. 

![X-ray pneumonia](../fig/xray-pneumonia.png){: width="600px"}

Compare the X-ray above with the following X-ray without the whiteness:

![X-ray normal](../fig/xray-normal.png){: width="600px"}

## Loading our dataset

We will train a neural network to detect chest X-rays with signs of pneumonia. To reduce the computational complexity of the task, we will use a collection of frontal-view X-rays that have been pre-processed to 28x28 pixels. We'll begin by loading the dataset and viewing the metadata.

```python
import medmnist

dataset = 'chestmnist'
metadata = medmnist.INFO[dataset]
print(metadata)
```

```
{'MD5': '02c8a6516a18b556561a56cbdd36c4a8',
 'description': 'The ChestMNIST is based on the NIH-ChestXray14 dataset, a dataset comprising 112,120 frontal-view X-Ray images of 30,805 unique patients with the text-mined 14 disease labels, which could be formulized as a multi-label binary-class classification task. We use the official data split, and resize the source images of 1×1024×1024 into 1×28×28.',
 'label': {'0': 'atelectasis',
  '1': 'cardiomegaly',
  '10': 'emphysema',
  '11': 'fibrosis',
  '12': 'pleural',
  '13': 'hernia',
  '2': 'effusion',
  '3': 'infiltration',
  '4': 'mass',
  '5': 'nodule',
  '6': 'pneumonia',
  '7': 'pneumothorax',
  '8': 'consolidation',
  '9': 'edema'},
 'license': 'CC0 1.0',
 'n_channels': 1,
 'n_samples': {'test': 22433, 'train': 78468, 'val': 11219},
 'python_class': 'ChestMNIST',
 'task': 'multi-label, binary-class',
 'url': 'https://zenodo.org/record/5208230/files/chestmnist.npz?download=1'}
```
{: .output}

Now let's download the dataset and view an example.


```python
from matplotlib import pyplot as plt

train_dataset = medmnist.ChestMNIST(split='train', download=True)
test_dataset = medmnist.ChestMNIST(split='test', download=True)
sample_image = train_dataset.imgs[0]

plt.imshow(sample_image)
plt.show()
```

![X-ray normal](../fig/mnist-sample.png){: width="300px"}

Viewing the content of the `sample_image` variable, we can see that the image is a rendering of a 28x28 array of numerical values that represent a colour density at each pixel.

```python
print(sample_image)
```

```
array([[ 61,   9,  13,  12,  12,  12,  13,  19,  31,  45,  70, 103, 123,
        143, 162, 162, 162, 143, 117,  97,  64,  40,  21,  13,  11,  13,
         21,  22],
       [ 54,   7,  10,  12,  20,  34,  48,  62,  74,  83,  92, 116, 131,
        145, 163, 162, 164, 148, 131, 115,  93,  80,  65,  44,  24,  21,
         29,  34],
       ...
```
{: .output}

## Data preparation

Now that we have our raw data, we need to prepare it for inputting into our (yet to be created!) neural network. 

[TODO: introduce tensorflow and concept of tensor]

First we'll convert our data to a TensorFlow `Dataset` and normalise the values on 0-1 scale. `train_dataset.imgs` contains our images and ` train_dataset.labels` contains the associated labels.

```python
import tensorflow as tf

# convert training set to a TensorFlow Dataset
train = tf.data.Dataset.from_tensor_slices((train_dataset.imgs, train_dataset.labels))

# convert test set to a TensorFlow Dataset
test = tf.data.Dataset.from_tensor_slices((test_dataset.imgs, test_dataset.labels))
```

## Creating the neural network

Instead of coding the network from scratch, we will use building blocks from the TensorFlow library to create a straightforward, sequential network. Our model will comprise of a stack of three layers: an input layer that takes our 28x28 set of features, a hidden layer with 10 neurons, and an output layer with the number of neurons matching our classes.

We are using "Dense" layers, one of the most commonly used types of layer. By dense, we mean that each layer in the neural network receives input from all neurons in the previous layer (hence their alternative name, the fully-connected layer).

[TODO: introduce softmax]

```python
# Create a `Sequential` model with a single hidden layer
model = tf.keras.models.Sequential()

# Add an input layer
input_shape = (28, 28)
model.add(tf.keras.Input(shape=input_shape))

# Add hidden layer
n_hidden_layer_1 = 10
model.add(tf.keras.layers.Dense(n_hidden_layer_1, activation=tf.nn.relu))

# Add an output layer
n_output = len(train_dataset.info['label'])
model.add(tf.keras.layers.Dense(n_output, activation=tf.nn.softmax))
```

## Defining the loss function and optimizer

[TODO: introduce loss function and optimizer]


```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

## Train the model

We now call the `fit()` method on the training data to train our network. 

```python
model.fit(train,
    epochs=10,
)
```

## Applying the model

Let's now try applying our trained model to unseen data to see how it performs on cases with known labels.

```python
# get a single image and label from the test set.
unseen_image = test.img[0]
unseen_label = test.label[0]

# predict the class using the network
prediction = model.predict(unseen_image)
print(prediction)
```

## Evaluation

As a more robust test, let's evaluate the model on our full test set.

```python
# evaluate

```

{% include links.md %}

