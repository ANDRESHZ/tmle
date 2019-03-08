# tmle

This repository will guide you through the process of building an image classifier using the classic computer vision approach and transfer learning (using architectures such as ResNet, DenseNet, etc.).

Models were built on images which came from ten classes. The below plot shows randomly selected images from each class (plot was created with `plot_grid_of_images` from [`tmle.visualizations`](https://github.com/stasulam/tmle/blob/master/tmle/visualizations.py)).

<p align="center">
    <img src="notebooks/figures/images_grid.png" width="750">
</p>

Description of methods used to classify above images has been divided into separate notebooks.

# Notebooks

* [Shallow classifier](https://github.com/stasulam/tmle/blob/master/notebooks/01_shallow_classifier.ipynb) presents a method of training SVM (with linear kernel) with histogram of oriented gradients used as a features. Choosing the best model is done by optimizing the parameters of the whole `Pipeline`. The algorithm used for hyperparameter tuning is Tree Parzen Estimator [1].
* [Dimension reduction](https://github.com/stasulam/tmle/blob/master/notebooks/02_dimension_reduction.ipynb) presents dimensionality reduction techniques such as PCA, t-SNE [2] and UMAP [3]. We have shown how the hyperparameters of these techniques affect the resulting visualizations. We explored the possibility of usage of outlier detection methods as a way of finding incorrectly labeled images.
* [Transfer learning](https://github.com/stasulam/tmle/blob/master/notebooks/03_transfer_learning.ipynb) presents methods of representing images as feature vectors obtained from Convolutional Neural Networks trained on different datasets. We have built SVM on top of feature vectors obtained from `ResNet18` and `ResNet50` (which were pretrained on `ImageNet`). Similarly to the shallow classifier, we optimized *hyperparameters* with TPE, but this time we performed much more experiments. Then, we have finetuned `ResNet18` with two different learning strategies. First, we used early stopping and manually modified learning rate and momentum parameters to obtain a well-performing network (avoiding overfitting). Next, we checked the capabilities of the recently published `AdaBound` [4]. Unfortunately, this gave worst results than first strategy.

Results (on test set):

| Model                      | Balanced accuracy score |
| ---------------------------|------------------------:|
| SVM (HOG features)         |0.3320                   |
| SVM (ResNet18 features)    |0.8020                   | 
| SVM (ResNet50 features)    |0.8450                   |
| ResNet18 (first strategy)  |0.8940                   |
| ResNet18 (second strategy) |0.8630 (overfitting)     |

Further works:

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

Then, install the package.
```bash
pip install git+https://github.com/stasulam/tmle.git
```

In order to use `tmle` on Google Colab use:
```bash
!pip install git+https://github.com/stasulam/tmle.git
!pip install pillow==4.1.1
```

## Summary



* `dataloaders` module implements methods which helps to iterate over images stored in directories which corresponds to class membership, sample *mini-batches* of images used during training and loading all images to memory at once.
* `model_selection` module implements methods which supports search for optimal *hyperparameters* of given classifier. It helps in storing results of optimization which can be reused in other experiments.
* `models` module ... .
* `transformers` module implements methods which allow to extract features from pretrained Convolutional Neural Networks (from `torchvision.models` or custom `CNN` trained with `torch`). It allows to use *histogram of oriented gradients* as part of `sklearn.Pipeline`.

***References:***

1. Bergstra, James S., et al. “Algorithms for hyper-parameter optimization”. Advances in Neural Information Processing Systems. 2011.
2. L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research, 2008.
3. UMAP
4. AdaBound
