---
title: "Explainability"
teaching: 20
exercises: 10
questions:
- "What is a saliency map?"
- "What aspects of an image contribute to predictions?"
objectives:
- "Review model performance with saliency maps."
keypoints:
- "Saliency maps are a popular form of explainability for imaging models."
- "Saliency maps should be used cautiously."
---

## Explainability

If a model is making a prediction, many of us would like to know how the decision was reached. Saliency maps - and related approaches - are a popular form of explainability for imaging models. 

Saliency maps use color to illustrate the extent to which a region of an image contributes to a given decision. Let's plot some saliency maps for our model:


```python
# !pip install tf_keras_vis
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

import numpy as np
from matplotlib import pyplot as plt
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore

gradcam = GradcamPlusPlus(best_model, clone=True)

def plot_map(cam, classe, prediction, img):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(np.squeeze(img), cmap='gray')
    axes[1].imshow(np.squeeze(img), cmap='gray')
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    i = axes[1].imshow(heatmap,cmap="jet",alpha=0.5)
    fig.colorbar(i)
    plt.suptitle("Class: {}. Pred = {}".format(classe, prediction))
    
for image_id in range(10):
    SEED_INPUT = dataset_test[image_id]
    CATEGORICAL_INDEX = [0]

    layer_idx = 18
    penultimate_layer_idx = 13
    class_idx  = 0

    cat_score = labels_test[image_id]
    cat_score = CategoricalScore(CATEGORICAL_INDEX)
    cam = gradcam(cat_score, SEED_INPUT, 
                  penultimate_layer = penultimate_layer_idx,
                  normalize_cam=True)
    
    # Display the class
    _class = 'normal' if labels_test[image_id] == 0 else 'effusion'

    _prediction = best_model.predict(dataset_test[image_id][np.newaxis,:,...], verbose=0)

    plot_map(cam, _class, _prediction[0][0], SEED_INPUT)
```
{: .language-python}

![Saliency maps](../fig/saliency.png){: width="600px"}

## Sanity checks

While saliency maps may offer us interesting insights about regions of an image contributing to a model's output, there are suggestions that this kind of visual assessment can be misleading. For example, the following abstract is from a paper entitled "[Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)":

> Saliency methods have emerged as a popular tool to highlight features in an input deemed relevant for the prediction of a learned model. Several saliency methods have been proposed, often guided by visual appeal on image data. In this work, we propose an actionable methodology to evaluate what kinds of explanations a given method can and cannot provide. We find that reliance, solely, on visual assessment can be misleading. Through extensive experiments we show that some existing saliency methods are independent both of the model and of the data generating process. Consequently, methods that fail the proposed tests are inadequate for tasks that are sensitive to either data or model, such as, finding outliers in the data, explaining the relationship between inputs and outputs that the model learned, and debugging the model. We interpret our findings through an analogy with edge detection in images, a technique that requires neither training data nor model.

The authors present the following comparison of the output of standard saliency methods with those of an edge detector. They explain that "the edge detector does not depend on model or training data, and yet produces results that bear visual similarity with saliency maps. This goes to show that visual inspection is a poor guide in judging whether an explanation is sensitive to the underlying model and data."

![Saliency maps](../fig/saliency_methods_and_edge_detector.png){: width="800px"}


{% include links.md %}
 



