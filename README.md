## ML Notes:

### Cross entropy:

A measure of the "difference" between two probability distributions.
For a true distribution `p`, we have
**1** as the the true class probability and **0** for all other classes.
`q` is the predicted probability from the model.

The mathematical definition for it could be:

$$ H(p, q) = -\sum\_{i=0}^{N-1} p(i) \ln(q(i)) $$

where

- we are summing over a series of terms each corresponding to a different category/class `i`,
  from `i = 0` to `i = N -1`.
- `p(i)` and `q(i)` are the true and model-predicted probability of category `i`, respectively.

Cross entropy will be low if true and predicted probabilities are close.
Otherwise, it will be higher.

### One-hot encoded arrays

For each value, a binary vector of all zeros except for a one at the index corresponding to the value is created. Good for representing categorical data.
For example, in a problem where you have three classes (0, 1, and 2), the one-hot encoding would be:

```
Class 0: [1, 0, 0]
Class 1: [0, 1, 0]
Class 2: [0, 0, 1]
```

### Categorical data:

Variables that can take on one of a limited, and usually fixed, number of possible values.
Example: education level - categories could be high school, bachelor's degree, master's degree, etc.

### Training, validation and test sets:

In machine learning, it's a common practice to divide the dataset into these three parts. Each set has a distinct purpose in the process of developing and evaluating the models.

1. **Training Set**:

   - The training set is used to train the model, i.e., to learn the parameters (e.g., weights in a neural network) that minimize the loss function. Here, the model "learns and adjusts its understanding"

2. **Validation Set**:

   - The validation set is used to tune hyperparameters (e.g., learning rate, model complexity) and to provide an unbiased evaluation of a model fit during the training phase. The model sees the validation set but doesn’t learn from it.
     Here, refines its learning strategy.

3. **Test Set**:
   - The test set is used to provide an unbiased evaluation of a final model fit. The model has never seen the test set during its training or model tuning phases.
     Here, the model is evaluated on its performance.

Each of these sets usually have a feature array and a label array. For instance, in the code snippet below:

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

the `x` arrays contain serialized data (in NumPy arrays) of images of handwritten digits, and the `y` arrays contain the labels/answers (0-9) corresponding to the images.

### Data normalization:

Process that changes the range of values of numeric data. Is done before training the model as it helps the model to converge faster. For example, here:

```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

the data is normalized by dividing it by 255.0 (the maximum value of a pixel). This way, the values of the pixels will be in the range `[0, 1]`.

:warning: changes the data, but not the relationship between them.

### Flattening:

Converting multi dimensional arrays to 1D because some layers expect a flat input.

### Overfitting:

When the model learns from detail and noise in the training data usually performing great in training data and very poorly in new data. The model "memorizes" the data, rather than learning/generalizing from it. Often a result of too complex models for simple data.

### Regularization:

Adds a penalty on different parameters reducing the model's freedom.
This avoids overfitting, but too much may underfit the model. Common ones are L1, L2 and dropout.

### Loss/cost function:

Measures how well the model's predictions match the true labels during training.

It is important because it guides the learning process by quantifying
the model's error to the optimization algorithm (eg. gradient descent)
This way, the model can change its parameter weights and improve itself.

The model should minimize this function.
