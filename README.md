# SoC (Physics Informed Neural Networks) - short summary of topics covered

## Week 1
* Basics of Python
* Essential libraries (numpy, pandas)
* Resources used : official documentations, problems to get familiar with syntax and methods (codes provided above)

## Weeks 2 and 3
* Basics of Neural Networks
* Learning to build a neural network from scratch using numpy
* Topics covered :
  - Neurons and Layers: Built foundational understanding of artificial neurons—computational units that take weighted inputs, apply a bias, and pass the result through an activation function. Constructed fully connected layers by stacking multiple neurons.

  - Forward Pass: Implemented the process of computing the output of a neural network by propagating inputs through the layers. Used matrix operations to efficiently handle multiple inputs at once.

  - Activation Functions: Explored the role of activation functions in introducing non-linearity into the network. Implemented and tested common functions like ReLU (rectified linear unit), Sigmoid and Softmax to better understand their mathematical behavior and use cases (ReLU - mainly for introducing non-linearity, sigmoid and SoftMAX - for estimating probabilities for classification taske).

  - Loss Functions: Studied how neural networks quantify prediction error using loss functions. Implemented Categorical Cross-Entropy for classification tasks.

  - Derivatives and Chain Rule: Developed an understanding of how gradients are computed using the chain rule from calculus, enabling the propagation of error backward through the network.

  - Backpropagation: Programmed backpropagation from scratch, computing partial derivatives for each layer to update weights and biases based on the loss gradient.
 
  - Gradient Descent: Learned how networks improve predictions by minimizing loss through optimization. Implemented basic gradient descent to update weights using calculated gradients.

  - Optimizers: Introduced advanced optimization techniques like Momentum, RMSProp and Adam to accelerate and stabilize training by adapting learning rates and smoothing updates.

  - Regularization: Studied about L1 (Lasso) and L2 (Ridge) regularization techniques to prevent overfitting by penalizing large weights during training.

* Resource : [Neural Networks from Scratch in Python by Harrison Kinsley & Daniel Kukieła (covered till chapter 14)](http://103.203.175.90:81/fdScript/RootOfEBooks/E%20Book%20collection%20-%202024%20-%20G/CSE%20%20IT%20AIDS%20ML/Neural%20Network.pdf)



## Week 4 onwards
* PyTorch
* Topics covered :
  - Introduction to PyTorch: Transitioned from manual NumPy implementations to PyTorch, a deep learning framework that simplifies model development through pre-defined classes and methods, as well as GPU acceleration.

  - Autograd: Explored PyTorch’s autograd system, which automatically computes gradients during the backward pass, eliminating the need for manual derivative calculations.

  - nn.Module: Learned to define custom neural networks by subclassing torch.nn.Module, organizing model architecture and parameters in a structured and reusable way.

  - Dataset and DataLoader: Used torch.utils.data.Dataset and DataLoader to manage large datasets efficiently—automating batching, shuffling, and parallel loading during training.
 
  - Neural Network Pipeline: Built a complete training workflow involving model instantiation, forward pass after segregating the training data into batches, loss computation (using predefined loss functions like CrossEntropyLoss), optimizer setup (e.g., Adam, SGD), and backward propagation using loss.backward() and optimizer.step().

* Resource : [PyTorch playlist by CampusX](https://www.youtube.com/watch?v=mDsFsnw3SK4&list=PLKnIA16_Rmvboy8bmDCjwNHgTaYH2puK7&index=2)

## Week 5
* PINNs
* Basic introduction to Physics Informed Neural Networks
* Learned the fundamental concept of Physics-Informed Neural Networks (PINNs), a powerful approach that integrates physical laws (typically expressed as differential equations) directly into the neural network training process.
* Forward Problem: Predict the solution of a physical system (e.g., temperature, displacement) given known parameters and initial/boundary conditions.\
What PINNs do: Learn a function that satisfies the governing DE and conditions using a neural network.
* Inverse Problem: Infer unknown parameters or functions in a DE (e.g., source term, diffusivity) from observed data.\
What PINNs do: Simultaneously learn the solution and unknown parameters by minimizing DE residuals and fitting data.
* Resource: [Lecture by Benjamin Moseley](https://www.youtube.com/watch?v=G_hIppUWcsc&t=1781s )

## Week 6
* PINNs using Tensorflow (another powerful deep learning framework)
* Built Physics-Informed Neural Networks using TensorFlow to solve differential equations. Learned to construct neural network architectures with tf.keras, define physics-based loss functions incorporating PDE residuals, and train models using automatic differentiation (tf.GradientTape). Implemented forward problems and experimented with inverse problems, gaining hands-on experience in leveraging TensorFlow’s GPU acceleration for PINNs.
* Resource : [YouTube playlist by elastropy](https://www.youtube.com/watch?v=pq3aAWU6kBQ&list=PLM7DTyYjZGuLmg3f6j40fEF18jyQmYsC2) (only Tensorflow part)

## Weeks 7 and 8
* Practising problems and working on the final implementation
* Solved certain classical Physics problems using PINNs (namely 1D and 2D heat equations).
* Worked on a PINN that could solve the 2D Navier-Stokes Equation as the final implementation project.

Note: Description and explainations of individual topics have been included in the python notebooks.
