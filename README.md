# Fashion MNIST
Simple deep convolutional neural network using the keras fashion_mnist dataset

## The network
The networked used for this project is as follows. All activations are relu except for the last Dense layer.
- Conv2D 32 3,3
- Conv2D 64 3,3
- MaxPooling2D 2,2
- Dropout .2
- Conv2D 64 3,3
- Conv2D 128 5,5
- MaxPooling2D 2,2
- Dropout .2
- Flatten
- Dense 32
- Dropout .4
- Dense 10 softmax

We use an Adam optimizer with amsgrad activated.

## Requirements for the project
- Python 3.5
- Keras
- Matplotlib
- Numpy
- Tensorflow (gpu is recommanded for optimal computation time)

> This project is using PyCharm as its IDE, hence the multiple files included alongside the main fashion_mnist.py file.

For more information on this type of network, please visit [the keras documentation](https://keras.io)
