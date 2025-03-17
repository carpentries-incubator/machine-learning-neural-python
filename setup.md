---
title: Setup
---

## Overview

This lesson is designed to be run on a personal computer.
All of the software and data used in this lesson are freely available online,
and instructions on how to obtain them are provided below.

## Install Python

In this lesson, we will be using Python 3 with some of its most popular scientific libraries.
Although one can install a plain-vanilla Python and all required libraries by hand, we recommend installing [Anaconda][anaconda-website],
a Python distribution that comes with everything we need for the lesson.
Detailed installation instructions for various operating systems can be found
on The Carpentries [template website for workshops][anaconda-instructions]
and in [Anaconda documentation][anaconda-install].

## Obtain lesson materials

The data that we are going to use for this project consists of 700 chest X-rays. These X-rays are a subset of the public NIH ChestX-ray dataset.

> Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers, ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471, 2017

1. Download [chest_xrays.zip](https://github.com/carpentries-incubator/machine-learning-neural-python/raw/gh-pages/data/chest_xrays.zip?download=).
1. Create a folder called `carpentries-ml-neural` on your Desktop.
3. Move downloaded files to `carpentries-ml-neural`.

## Launch Python interface

To start working with Python, we need to launch a program that will interpret and execute our Python commands. Below we list several options. If you don't have a preference, proceed with the top option in the list that is available on your machine. Otherwise, you may use any interface
you like.

## Option A: Jupyter Notebook

A Jupyter Notebook provides a browser-based interface for working with Python.
If you installed Anaconda, you can launch a notebook in two ways:

> ## Anaconda Navigator
>
> 1. Launch Anaconda Navigator.
> It might ask you if you'd like to send anonymized usage information to Anaconda developers:
> ![Anaconda Navigator first launch](
{{ page.root }}{% link fig/anaconda-navigator-first-launch.png %})
> Make your choice and click "Ok, and don't show again" button.
> 2. Find the "Notebook" tab and click on the "Launch" button:
> ![Anaconda Navigator Notebook launch](
{{ page.root }}{% link fig/anaconda-navigator-notebook-launch.png %})
> Anaconda will open a new browser window or tab with a Notebook Dashboard showing you the
> contents of your Home (or User) folder.
> 3. Navigate to the `data` directory by clicking on the directory names leading to it:
> `Desktop`, `swc-python`, then `data`:
> ![Anaconda Navigator Notebook directory](
{{ page.root }}{% link fig/jupyter-notebook-data-directory.png %})
> 4. Launch the notebook by clicking on the "New" button and then selecting "Python 3":
> ![Anaconda Navigator Notebook directory](
{{ page.root }}{% link fig/jupyter-notebook-launch-notebook.png %})
{: .solution}

> ## Command line (Terminal)
>
> 1\. Navigate to the `data` directory:
>
> > ## Unix shell
> > If you're using a Unix shell application, such as Terminal app in macOS, Console or Terminal
> > in Linux, or [Git Bash][gitbash] on Windows, execute the following command:
> > ~~~
> > cd ~/Desktop/swc-python/data
> > ~~~
> > {: .language-bash}
> {: .solution}
>
> > ## Command Prompt (Windows)
> > On Windows, you can use its native Command Prompt program.  The easiest way to start it up is
> > pressing <kbd>Windows Logo Key</kbd>+<kbd>R</kbd>, entering `cmd`, and hitting
> > <kbd>Return</kbd>. In the Command Prompt, use the following command to navigate to
> > the `data` folder:
> > ~~~
> > cd /D %userprofile%\Desktop\swc-python\data
> > ~~~
> > {: .source}
> {: .solution}
>
> 2\. Start Jupyter server
>
> > ## Unix shell
> > ~~~
> > jupyter notebook
> > ~~~
> > {: .language-bash}
> {: .solution}
>
> > ## Command Prompt (Windows)
> > ~~~
> > python -m notebook
> > ~~~
> > {: .source}
> {: .solution}
>
> 3\. Launch the notebook by clicking on the "New" button on the right and selecting "Python 3"
> from the drop-down menu:
> ![Anaconda Navigator Notebook directory](
{{ page.root }}{% link fig/jupyter-notebook-launch-notebook2.png %})
{: .solution}

&nbsp; <!-- vertical spacer -->

## Option B: Cloud Notebook

Colaboratory, or "Colab", is a cloud service that allows you to run a Jupyter-like Notebook in a web browser. To open a notebook, visit the [Colaboratory website][google-colab]. You can upload your datasets using the "Files" panel on the left side of the page.

![Google Colab]({{ page.root }}{% link fig/colab_files.png %})

[anaconda-install]: https://docs.anaconda.com/anaconda/install
[anaconda-instructions]: https://carpentries.github.io/workshop-template/#python
[anaconda-website]: https://www.anaconda.com/
[gitbash]: https://gitforwindows.org
[eicu-data]: https://doi.org/10.13026/4mxk-na84
[google-colab]: https://colab.research.google.com/

{% include links.md %}
