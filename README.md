# tmle

This repository will guide you through the process of building an image classifier using the classic computer vision approach and transfer learning (using architectures such as ResNet, DenseNet, etc.).

# Notebooks

* [`01_shallow_classifier`]() presents a method of training SVM (with linear kernel due to the performance issues) on top of histograms of oriented gradients. Choosing the best model is done by optimizing the parameters of the whole `Pipeline` with the use of a Tree Parzen Estimator [1].
* [`02_dimension_reduction`]() presents ... .
* [`03_transfer_learning`]() presents ... .

References:

[1] Bergstra, James S., et al. “Algorithms for hyper-parameter optimization”. Advances in Neural Information Processing Systems. 2011.
