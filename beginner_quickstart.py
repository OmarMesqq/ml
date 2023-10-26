# Activity from: https://www.tensorflow.org/tutorials/quickstart/beginner
# Image classification problem - multi class classification

import tensorflow as tf

# Load the MNIST database of handwritten digits
mnist = tf.keras.datasets.mnist

# Get training and test set tuples
# Each of them have a feature and label array
# First one contains the serialized data (in NumPy arrays)
# for training and testing
# Second one has the answers for both sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize the data
# Set the data values between 0 and 1 which usually helps the 
# model learn faster/better. It changes the data, but NOT
# the relationship between them
x_train, x_test = x_train / 255.0, x_test / 255.0


# Create a linear stack of layers
# Here, each layer only receives one tensor
# from the previous layer and sends only one tensor  to next 
#
# Pass a [list] of layer instances:
#   First one flattens the input data expecting
#       28x28 array of pixel values
#       Flattening: converts multi dimensional arrays to 1D 
#       because some layers expect a flat input
#
#   Second one is a fully connected layer, i.e 
#       each neuron if the 128 ones 
#       is connected to every neuron in the previous layer
#
#   Third one regularizes 20% of the input units to 0 during training to avoid overfitting
#       Regularization: adds a penalty on different parameters reducing the model's
#           freedom. This avoids overfitting, but too much may underfit the model
#           Common ones are L1, L2 and dropout        
#       Overfitting: the model learns from detail and noise in the training data
#           usually performing great in training data and very poorly in new one
#           The model "memorizes" the data, rather than learning/generalizing from it
#           Often a result of too complex models for simple data
#
#   Lastly another Dense layer with linear (no activation) by default
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Create a loss/cost function
# measures how well the model's predictions match the true labels during training
# The model should minimize this function
#
# It is important because it guides the learning process by quantifying 
# the model's error to the optimization algorithm (eg. gradient descent)
# This way, the model can change its parameter weights and improve itself

# Computes crossentropy loss from raw logits (instead of probabilities
# from softmax for instance)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# Configure/compile the model
# using our defined loss function, 
# Adam optimization (stochastic gradient descent) - has
# adaptive learning rate properties, which should converge faster
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# Train the model 
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
# Check model performance ideally with 
# a validation set, but we only have test set
model.evaluate(x_test,  y_test, verbose=2)

# Convert the raw logits (any real number) to probabilities (float between 0 and 1)
# We'll use Softmax
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


# ! Let's see the prediction for the initial 5 test images
print(probability_model(x_test[:5]))
