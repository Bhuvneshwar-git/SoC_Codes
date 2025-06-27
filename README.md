# SoC (Physics Informed Neural Networks) - short summary of topics covered

## Week 1
* Basics of Python
* Essential libraries (numpy, pandas)
* Resources used : official documentations, problems to get familiar with syntax and methods (codes provided above)

## Weeks 2 and 3
* Basics of Neural Networks
* Learning to build a neural network from scratch using numpy
* Topics covered :
  - Neurons and Layers: Implemented single neurons and layered networks using NumPy; understood how inputs are transformed through weights and biases.
  - Forward Pass: Calculated outputs layer by layer using matrix operations; understood basic flow of data through the network.
  - Activation Functions: Studied and implemented ReLU, Sigmoid, and Softmax to introduce non-linearity and interpret outputs as probabilities.
  - Loss Functions: Studied about the Categorical Cross-Entropy for classification tasks and MSE for regression tasks.
  - Derivatives and Chain Rule: Calculated gradients using calculus, applying the chain rule across layers.
  - Backpropagation: Implemented backpropagation to compute weight updates based on the error gradient flowing backward.
  - Gradient Descent: Learned the optimization process for updating weights using gradients to minimize the loss function.
  - Optimizers: Explored Momentum, RMSProp, and Adam optimizers to enhance and accelerate convergence.
  - Regularization: Implemented L1 and L2 regularization techniques to reduce overfitting by penalizing large weights.
 
Note: Description and explainations of individual topics have been included in the python notebook.

* Resource : Neural Networks from Scratch in Python by Harrison Kinsley & Daniel Kukieła (covered till chapter 14)



## Week 4 onwards
* PyTorch
* Topics covered :
  - Introduction to PyTorch: Began working with the PyTorch framework for building and training neural networks more efficiently.

  - Autograd: Learned PyTorch’s automatic differentiation system to compute gradients dynamically during backpropagation.

  - nn.Module: Explored how to define custom neural network architectures by subclassing nn.Module.

  - Neural Network Pipeline: Implemented the full training pipeline including model definition, loss computation, optimizer usage, and training loop.

  - Dataset and DataLoader: Learned how to implement abd utilize PyTorch’s utilities to load and batch data efficiently for training.
 
Note: Description and explainations of individual topics have been included in the python noteboos.

* Resource : PyTorch playlist by CampusX
