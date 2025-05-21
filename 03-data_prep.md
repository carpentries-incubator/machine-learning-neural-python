---
title: Data preparation
teaching: 40
exercises: 20
---

::::::::::::::::::::::::::::::::::::::: objectives

- Split the dataset into training, validation, and test sets.
- Prepare image and label arrays in the format expected by TensorFlow.
- Apply basic image augmentation to increase training data diversity.
- Understand the role of data preprocessing in model generalization.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- Why do we divide data into training, validation, and test sets?
- What is data augmentation, and why is it useful for small datasets?
- How can random transformations help improve model performance?

::::::::::::::::::::::::::::::::::::::::::::::::::


## Partitioning the dataset

Before training our model, we must split the dataset into three subsets:

- **Training set**: Used to train the model.
- **Validation set**: Used to tune parameters and monitor for overfitting.
- **Test set**: Used for final performance evaluation.

This separation helps ensure that our model generalizes to new, unseen data.

To ensure reproducibility, we set a `random_state`, which controls the random number generator and guarantees the same split every time we run the code.

TensorFlow expects image input in the format:  

`[batch_size, height, width, channels]`  

So we’ll also expand our image and label arrays to include the final channel dimension (grayscale images have 1 channel).


```python
from sklearn.model_selection import train_test_split

# Reshape arrays to include a channel dimension:
# [height, width] → [height, width, 1]
dataset_expanded = dataset[..., np.newaxis]
labels_expanded = labels[..., np.newaxis]

# Create training and test sets (85% train, 15% test)
dataset_train, dataset_test, labels_train, labels_test = train_test_split(
    dataset_expanded, labels_expanded, test_size=0.15, random_state=42)

# Further split training set to create validation set (15% of remaining data)
dataset_train, dataset_val, labels_train, labels_val = train_test_split(
    dataset_train, labels_train, test_size=0.15, random_state=42)

print("No. images, x_dim, y_dim, colors) (No. labels, 1)\n")
print(f"Train: {dataset_train.shape}, {labels_train.shape}")
print(f"Validation: {dataset_val.shape}, {labels_val.shape}")
print(f"Test: {dataset_test.shape}, {labels_test.shape}")
```

```output
No. images, x_dim, y_dim, colors) (No. labels, 1)

Train: (505, 256, 256, 1), (505, 1)
Validation: (90, 256, 256, 1), (90, 1)
Test: (105, 256, 256, 1), (105, 1)
```

## Data Augmentation

Our dataset is small, which increases the risk of **overfitting**, when a model learns patterns specific to the training set but performs poorly on new data.

**Data augmentation** helps address this by creating modified versions of the training images on-the-fly using random transformations. This teaches the model to become more robust to variations it might encounter in real-world data.

We can use `ImageDataGenerator` to define the types of augmentation to apply.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define what kind of transformations we would like to apply
# such as rotation, crop, zoom, position shift, etc
datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    zoom_range=0,
    horizontal_flip=False)
```

:::::::::::::::::::::::::::::::::::::::  challenge

## Exercise

A) Modify the `ImageDataGenerator` to include one or more of the following:

- `rotation_range=20`
- `zoom_range=0.2`
- `horizontal_flip=True`

:::::::::::::::  solution

## Solution

A) Here's an example:

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

Now let's view the affect on our X-rays!:

```python
# specify path to source data
path = os.path.join("chest_xrays")
batch_size=5

val_generator = datagen.flow_from_directory(
        path, color_mode="rgb",
        target_size=(256, 256),
        batch_size=batch_size)

def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img.astype('uint8'))
    plt.tight_layout()
    plt.show()

augmented_images = [val_generator[0][0][0] for i in range(batch_size)]
plot_images(augmented_images)
```

![](fig/xray_augmented.png){alt='X-ray augmented' width="1200px"}

:::::::::::::::::::::::::::::::::::::::  challenge

## Exercise

A) How do the new augmentations affect the appearance of the X-rays?  
Can you still tell they are chest X-rays?

:::::::::::::::  solution

## Solution

A) The augmented images may appear rotated, zoomed, or flipped.  
While they might look distorted, they remain visually recognizable as chest X-rays. These augmentations help the model generalize better to real-world variability.

In medical imaging, always consider clinical context. Some transformations, like left-right flipping, could lead to anatomically incorrect inputs if not handled carefully.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

Now we have some data to work with, let's start building our model.


:::::::::::::::::::::::::::::::::::::::: keypoints

- Data should be split into separate sets for training, validation, and testing to fairly evaluate model performance.
- TensorFlow expects input images in the shape (batch, height, width, channels).
- Data augmentation increases the variety of training data by applying random transformations.
- Augmented images help reduce overfitting and improve generalization to new data.

::::::::::::::::::::::::::::::::::::::::::::::::::



