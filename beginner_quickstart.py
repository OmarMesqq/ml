# Activity from: https://www.tensorflow.org/tutorials/quickstart/beginner
# Image classification problem - multi class classification

import tensorflow as tf

# Load the MNIST database of handwritten digits
mnist = tf.keras.datasets.mnist

# Get training and test set tuples
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0


# Create a linear stack of layers:
# each layer only receives one tensor from the previous layer and sends only one tensor to next 

# Inside the `Sequential` we are passing a list of layer instances:
#   First one flattens the input data expecting 28x28 array of pixel values
#
#   Second one is a fully connected layer, i.e each neuron of the 128 ones 
#   is connected to every neuron in the previous layer
#
#   Third one regularizes 20% of the input units to 0        
#
#   Lastly another Dense layer with linear (no activation) by default
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Create a loss/cost function
# In this case, computes cross entropy loss from raw logits (instead of probabilities)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# Configure/compile the model
# using our defined loss function and
# Adam optimization (stochastic gradient descent) 
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Train the model 
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test,  y_test, verbose=2)

# Convert the raw logits (any real number) to probabilities (float between 0 and 1)
# by adding a softmax layer
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Make predictions on the first 5 test images
predictions = probability_model(x_test[:5])

# Get the predicted labels
predicted_labels = tf.argmax(predictions, axis=1)


print(f'True labels: {y_test[:5]}')
print(f'Predicted labels: {predicted_labels.numpy()}')
