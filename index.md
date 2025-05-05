---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---

This lesson gives an introduction to artificial neural networks. We begin by an outlining an important application of machine learning in healthcare: the development of algorithms for classification of chest X-ray images. During the lesson we explore how to prepare and visualise data for algorithm development, and construct a neural net that is able to classify disease.

### Other related lessons
#### Introduction to deep learning
The [Introduction to to deep learning lesson](https://carpentries-incubator.github.io/deep-learning-intro/)
is a more general introduction to deep learning, applying deep learning to different domains.
Whereas this lesson focuses on the application of deep learning on medical images. 
If you are interested in other domains than medical images you could choose to follow that lesson.

#### Introduction to machine learning in Python with scikit-learn
The [Introduction to machine learning in Python with scikit-learn lesson](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/)
introduces practical machine learning using Python. It might be a good lesson to follow in preparation for this lesson,
since basic knowledge of machine learning and Python programming skills are required for this lesson.


<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}

> ## Prerequisites
>
> You need to understand the basics of Python before tackling this lesson. The lesson sometimes references Jupyter Notebook although you can use any Python interpreter mentioned in the [Setup][lesson-setup].
{: .prereq}

### Getting Started

To get started, follow the directions on the "[Setup][lesson-setup]" page to download data and install a Python interpreter.

[eicu-crd]: https://doi.org/10.13026/C2WM1R

{% include links.md %}
