# tmle

This repository will guide you through the process of building an image classifier using the classic computer vision approach and transfer learning (using architectures such as ResNet, DenseNet, etc.).

Models were built on images which came from ten classes. The below plot shows randomly selected images from each class (plot was created with `plot_grid_of_images` from [`tmle.visualizations`](https://github.com/stasulam/tmle/blob/master/tmle/visualizations.py)).

<p align="center">
    <img src="notebooks/figures/images_grid.png" width="750">
</p>

Data (features obtained from ResNet, t-SNE and UMAP embeddings) and models (shallow classifier, SVMs, CNNs, etc.) can be downloaded from [here](https://drive.google.com/open?id=1151gQZHJLFDiqhmOmiWMDpiT0IwWPGTZ).

Description of methods used to classify above images has been divided into separate notebooks.

# Notebooks

* [Shallow classifier](https://github.com/stasulam/tmle/blob/master/notebooks/01_shallow_classifier.ipynb) presents a method of training SVM (with linear kernel) with histogram of oriented gradients used as a features. Choosing the best model is done by optimizing the parameters of the whole `Pipeline`. The algorithm used for hyperparameter tuning is Tree Parzen Estimator [1].
* [Dimension reduction](https://github.com/stasulam/tmle/blob/master/notebooks/02_dimension_reduction.ipynb) presents dimensionality reduction techniques such as PCA, t-SNE [2, 3] and UMAP [4]. We have shown how the hyperparameters of these techniques affect the resulting visualizations. We explored the possibility of usage of outlier detection methods as a way of finding incorrectly labeled images.
* [Transfer learning](https://github.com/stasulam/tmle/blob/master/notebooks/03_transfer_learning.ipynb) presents methods of representing images as feature vectors obtained from Convolutional Neural Networks trained on different datasets. We have built SVM on top of feature vectors obtained from ResNet18 and ResNet50 (which were pretrained on `ImageNet`). Similarly to the shallow classifier, we optimized *hyperparameters* with TPE, but this time we performed much more experiments. Then, we have finetuned ResNet18 with two different learning strategies. First, we used early stopping and manually modified learning rate and momentum parameters to obtain a well-performing network (avoiding overfitting at the same time). Next, we checked the capabilities of the recently published `AdaBound` [5]. Unfortunately, this gave worst results than first strategy.

## Results

The following table shows the results obtained on the test set.

| Model                      | Balanced accuracy score |
| ---------------------------|------------------------:|
| SVM (HOG features)         |0.3320                   |
| SVM (ResNet18 features)    |0.8020                   | 
| SVM (ResNet50 features)    |0.8450                   |
| ResNet18 (SGD)             |0.8940                   |
| ResNet18 (AdaBound)        |0.8630 (overfitting)     |

## Further works

Promising directions:

* add `sampler` during loading images to mini-batches in order to measure the impact of class imbalanced.
* use outliers detection methods to remove incorrectly labeled images from dataset.
* try with different (deeper in case of ResNet) architectures. [Paper with code](https://paperswithcode.com/task/image-classification) is a good starting point.

# Package

The `tmle` package provides methods which are useful in training image classifiers.

## Installation

Create conda environment dedicated to the project.
```bash
conda create -n tmle python=3.6
```

Then, activate the environment and install the package.
```bash
source activate tmle
# activate tmle (on Windows)
pip install git+https://github.com/stasulam/tmle.git
```

In order to use `tmle` on Google Colab use:
```bash
!pip install git+https://github.com/stasulam/tmle.git
!pip install pillow==4.1.1
```

## Summary

The following is a brief summary of the functionalities of the individual modules:

* `dataloaders` module implements methods which are helpful in: (i) iterating over images stored in directories which correspond to a class membership, (ii) sampling *mini-batches* during training neural networks, (iii) loading all images to memory at once.
* `model_selection` module implements methods which are helpful in: (i) searching for optimial *hyperparameters* of given classifier or `Pipeline` (in case of a `Pipeline` we optimize not only a classifier, but also a preprocessing stage), (ii) preserving results of experiments with `hyperopt.Trials` object.
* `models` module implements manager which was used during *transfer learning* part of task.
* `transformers` module implements methods which allow to extract features from pretrained Convolutional Neural Networks (especially from `torchvision.models`, but they support all models which were implemented in `torch`). This module consists of a `transformer` which allow to make `skimage.feature.hog` a part of `sklearn.Pipeline` (`TODO`: add voting known from `opencv` implementation of `HOGDescriptor`).

***Resources:***

1. https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
2. https://lvdmaaten.github.io/tsne/
3. https://distill.pub/2016/misread-tsne/
4. https://github.com/lmcinnes/umap
5. https://github.com/Luolc/AdaBound
