# tmle

This repository will guide you through the process of building an image classifier using the classic computer vision approach and transfer learning (using architectures such as ResNet, DenseNet, etc.).

Models were built on images coming from ten classes. The plot below shows randomly selected images from each class.

<img src='notebooks/figures/images_grid.png' width=600 align='middle'>

The description of the methods used has been divided into separate notebooks.

# Notebooks

* [`01_shallow_classifier`]() presents a method of training SVM (with linear kernel due to the performance issues) on top of histograms of oriented gradients. Choosing the best model is done by optimizing the parameters of the whole `Pipeline` with the use of a Tree Parzen Estimator [1].
* [`02_dimension_reduction`]() presents dimensionality reduction techniques such as `PCA`, `t-SNE` and `UMAP`. We have shown how the hyperparameters of these techniques affect the resulting visualizations. We will then explore the possibility of using outlier detection methods (ie. Isolation Forest or {H}DBSCAN) to discover images with incorrectly assigned labels.
* [`03_transfer_learning`]() presents ... .

# Package

The *tmle* package provides methods useful (at least, in my opinion) in the problem of building an image classifier.

* `dataloaders` module implements methods which helps to iterate over images stored in directories which corresponds to class membership, sample *mini-batches* of images used during training and loading all images to memory at once.
* `model_selection` module implements methods which supports search for optimal *hyperparameters* of given classifier. It helps in storing results of optimization which can be reused in other experiments.
* `models` module ... .
* `transformers` module implements methods which allow to extract features from pretrained Convolutional Neural Networks (from `torchvision.models` or custom `CNN` trained with `torch`). It allows to use *histogram of oriented gradients* as part of `sklearn.Pipeline`.

***References:***

1. Bergstra, James S., et al. “Algorithms for hyper-parameter optimization”. Advances in Neural Information Processing Systems. 2011.
2. L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research, 2008.
3. 
