---
title: "Saliency maps"
teaching: 20
exercises: 10
questions:
- "What is a saliency map?"
- "How can I understand what aspects of an image contribute to predictions made by a network?"
objectives:
- "Review model performance with saliency maps."
keypoints:
- "We can use tools to highlight the areas of an image that contribute to a model output."
---

## Understanding more about how the network makes a prediction

We can use tools to rank the pixels in an image based on their contribution to the network prediction. Here we use GradCAM which looks at the output of the penultimate layer (that is the convolutional layer just before dense layers).

```python
# !pip install tf_keras_vis
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

import numpy as np
from matplotlib import pyplot as plt
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore

gradcam = GradcamPlusPlus(best_model,
                          clone=True)

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
    
    #Let's show which class it belongs to
    _classe = 'normal' if labels_test[image_id]==0 else 'effusion'

    _prediction = best_model.predict(dataset_test[image_id][np.newaxis,:,...], verbose=0)

    plot_map(cam, _classe, _prediction[0][0], SEED_INPUT)
```
{: .language-python}

![Saliency maps](../fig/saliency.png){: width="600px"}

{% include links.md %}
 



