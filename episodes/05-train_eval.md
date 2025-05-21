---
title: Training and evaluation
teaching: 20
exercises: 10
---

::::::::::::::::::::::::::::::::::::::: objectives

- Compile a neural network with a suitable loss function and optimizer.
- Train a convolutional neural network using batches of data.
- Monitor model performance during training using training and validation loss and accuracy.
- Evaluate a trained model on a held-out test set.

::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::: questions

- How is a neural network trained to make better predictions?
- What do training loss and accuracy tell us?
- How do we evaluate a model’s performance on unseen data?

::::::::::::::::::::::::::::::::::::::::::::::::::


## Compile and train your model

Now that the model architecture is complete, it is ready to be compiled and trained! The distance between our predictions and the true values is the error or "loss". The goal of training is to minimise this loss.

Through training, we seek an optimal set of model parameters. Using an optimization algorithm such as gradient descent, our model weights are iteratively updated as each batch of data is processed.

Batch size is the number of training examples processed before the model parameters are updated. An epoch is one complete pass through all of the training data. In an epoch, we use all of the training examples once.

```python
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the network optimization method. 
# Adam is a popular gradient descent algorithm
# with adaptive, per-parameter learning rates.
custom_adam = optimizers.Adam()

# Compile the model defining the 'loss' function type, optimization and the metric.
model.compile(loss='binary_crossentropy', optimizer=custom_adam, metrics=['acc'])

# Save the best model found during training
checkpointer = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss',
                               verbose=1, save_best_only=True)

# Now train our network!
# steps_per_epoch = len(dataset_train)//batch_size
hist = model.fit(datagen.flow(dataset_train, labels_train, batch_size=32), 
                 steps_per_epoch=15, 
                 epochs=10, 
                 validation_data=(dataset_val, labels_val), 
                 callbacks=[checkpointer])
```

We can now plot the results of the training. "Loss" should drop over successive epochs and accuracy should increase.

```python
plt.plot(hist.history['loss'], 'b-', label='train loss')
plt.plot(hist.history['val_loss'], 'r-', label='val loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

plt.plot(hist.history['acc'], 'b-', label='train accuracy')
plt.plot(hist.history['val_acc'], 'r-', label='val accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()
```

![](fig/training_curves.png){alt='Training curves' width="600px"}

:::::::::::::::::::::::::::::::::::::::  challenge

## Exercise

Examine the training and validation curves.

A) What does it mean if the training loss continues to decrease, but the validation loss starts increasing?  
B) Suggest two actions you could take to reduce overfitting in this situation.  
C) Bonus: Try increasing the dropout rate in your model. What happens to the validation accuracy?

:::::::::::::::  solution

## Solution

A) If the training loss decreases while the validation loss increases, the model is **overfitting** — it’s learning the training data too well and struggling to generalize to unseen data.

B) You could:
- **Increase regularization** (e.g. by raising the dropout rate)
- **Add more training data**
- **Use data augmentation**
- **Simplify the model** to reduce capacity

C) Increasing dropout may lower performance slightly but improve generalization. Always compare the training and validation accuracy/loss to decide.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::


## Evaluating your model on the held-out test set

In this step, we present the unseen test dataset to our trained network and evaluate the performance.

```python
from tensorflow.keras.models import load_model 

# Open the best model saved during training
best_model = load_model('best_model.keras')
print('\nNeural network weights updated to the best epoch.')
```

Now that we've loaded the best model, we can evaluate the accuracy on our test data.

```python
# We use the evaluate function to evaluate the accuracy of our model in the test group
print(f"Accuracy in test group: {best_model.evaluate(dataset_test, labels_test, verbose=0)[1]}")
```

```output
Accuracy in test group: 0.80
```



:::::::::::::::::::::::::::::::::::::::: keypoints

- Neural networks are trained by adjusting weights to minimize a loss function using optimization algorithms like Adam.
- Training is done in batches over multiple epochs to gradually improve performance.
- Validation data helps detect overfitting and track generalization during training.
- The best model can be selected by monitoring validation loss and saved for future use.
- Final performance should be evaluated on a separate test set that the model has not seen during training.

::::::::::::::::::::::::::::::::::::::::::::::::::


