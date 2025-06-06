---
title: Introduction
teaching: 20
exercises: 10
---

::::::::::::::::::::::::::::::::::::::: objectives

- Understand what chest X-rays show and how they are used in diagnosis.
- Recognize pleural effusion as a condition visible on chest X-rays.
- Gain familiarity with the NIH ChestX-ray dataset.
- Load and explore a balanced set of labeled chest X-rays for model training.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- What kinds of conditions can be detected in chest X-rays?
- How does pleural effusion appear on a chest X-ray?
- How can chest X-ray data be used to train a machine learning model?

::::::::::::::::::::::::::::::::::::::::::::::::::

## Chest X-rays

Chest X-rays are frequently used in healthcare to view the heart, lungs, and bones of patients. On an X-ray, broadly speaking, bones appear white, soft tissue appears grey, and air appears black. The images can show details such as:

- Lung conditions, for example pneumonia, emphysema, or air in the space around the lung.
- Heart conditions, such as heart failure or heart valve problems.
- Bone conditions, such as rib or spine fractures
- Medical devices, such as pacemaker, defibrillators and catheters. X-rays are often taken to assess whether these devices are positioned correctly.

In recent years, organisations like the [National Institutes of Health](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) have released large collections of X-rays, labelled with common diseases. The goal is to stimulate the community to develop algorithms that might assist radiologists in making diagnoses, and to potentially discover other findings that may have been overlooked.

The following figure is from a study by [Xiaosong Wang et al](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf). It illustrates eight common diseases that the authors noted could be be detected and even spatially-located in front chest x-rays with the use of modern machine learning algorithms.

![](fig/wang_et_al.png){alt='Chest X-ray diseases' width="600px"}

:::::::::::::::::::::::::::::::::::::::  challenge

## Exercise

A) What are some possible challenges when working with real chest X-ray data?  
Think about issues related to the data itself (e.g. image quality, labels), as well as how the data might be used in a clinical or machine learning setting.

:::::::::::::::  solution

## Solution

A) Possible challenges include:

- **Label noise**: Labels are often derived from radiology reports using automated tools, and may not be 100% accurate.
- **Ambiguity in diagnosis**: Even expert radiologists may disagree on the interpretation of an image.
- **Variability in image quality**: X-rays may be over- or under-exposed, blurry, or taken from non-standard angles.
- **Presence of confounders**: Images may include pacemakers, tubes, or other devices that distract or bias a model.
- **Data imbalance**: In real-world datasets, some conditions (like pleural effusion) may be much less common than others.
- **Generalization**: A model trained on one dataset may not perform well on data from a different hospital or population.

These challenges highlight why data curation, domain expertise, and robust validation are critical in medical machine learning.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

## Pleural effusion

Thin membranes called "pleura" line the lungs and facilitate breathing. Normally there is a small amount of fluid present in the pleura, but certain conditions can cause excess build-up of fluid. This build-up is known as pleural effusion, sometimes referred to as "water on the lungs".

Causes of pleural effusion vary widely, ranging from mild viral infections to serious conditions such as congestive heart failure and cancer. In an upright patient, fluid gathers in the lowest part of the chest, and this build up is visible to an expert.

For the remainder of this lesson, we will develop an algorithm to detect pleural effusion in chest X-rays. Specifically, using a set of chest X-rays labelled as either "normal" or "pleural effusion", we will train a neural network to classify unseen chest X-rays into one of these classes.

## Loading the dataset

The data that we are going to use for this project consists of 350 "normal" chest X-rays and 350 X-rays that are labelled as showing evidence pleural effusion. These X-rays are a subset of the public NIH ChestX-ray dataset.

> Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers, ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471, 2017

Let's begin by loading the dataset.

```python
# The glob module finds all the pathnames matching a specified pattern
from glob import glob
import os

# If your dataset is compressed, unzip with:
# !unzip chest_xrays.zip

# Define folders containing images
data_path = os.path.join("chest_xrays")
effusion_path = os.path.join(data_path, "effusion", "*.png")
normal_path = os.path.join(data_path, "normal", "*.png")

# Create list of files
effusion_list = glob(effusion_path)
normal_list = glob(normal_path)

print('Number of cases with pleural effusion: ', len(effusion_list)) 
print('Number of normal cases: ', len(normal_list))
```

```output
Number of cases with pleural effusion:  350
Number of normal cases:  350
```



:::::::::::::::::::::::::::::::::::::::: keypoints

- Chest X-rays are widely used to identify lung, heart, and bone abnormalities.
- Pleural effusion is a condition where excess fluid builds up around the lungs, visible in chest X-rays.
- Large public datasets like the NIH ChestX-ray dataset enable the development of machine learning models to detect disease.
- In this lesson, we will train a neural network to classify chest X-rays as either “normal” or showing pleural effusion.
- We begin by loading a balanced dataset of labeled chest X-ray images.

::::::::::::::::::::::::::::::::::::::::::::::::::


