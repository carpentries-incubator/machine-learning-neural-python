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

## Classifying pneumonia

We will train a neural network to detect chest X-rays with signs of pneumonia. To reduce the computational complexity of the task, we will use a collection of X-rays that have downsampled to 28x28 pixels.

The following image shows a downsampled X-ray with signs of pneumonia. 

![X-ray pneumonia 28x28px](../fig/xray-pneumonia-28-28.png){: width="300px"}

The following image shows a downsampled X-ray with no obvious signs of pneumonia.

![X-ray normal 28x28px](../fig/xray-normal-28-28.png){: width="300px"}

{% include links.md %}

