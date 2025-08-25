import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data: flatten the images and normalize pixel values
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

Y = x_train

X = np.random.rand(x_train.shape[0],x_train.shape[1])

# Define the MLP model using Keras Sequential API
model = Sequential([
    # Input layer and first hidden layer (input shape is now 28*28 = 784)
    Dense(units=128, activation='relu', input_shape=(784,)),
    # Second hidden layer
    Dense(units=164, activation='relu'),
    Dense(units=164, activation='relu'),
    Dense(units=164, activation='relu'),
    Dense(units=164, activation='relu'),
    # Output layer (output units is now 10 for the 10 digits)
    Dense(units=x_train.shape[1], activation='relu') # Use softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='sgd',
              loss='mae', # Use categorical crossentropy for multi-class
              metrics=['accuracy'])

# Train the model
epochs = 200 # Reduced epochs for a quicker example
batch_size = 100
history = model.fit(X, Y,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1, # Set verbose to 1 to see training progress
)


# Print loss and accuracy at the end of training
print(f"\nFinal Training Loss: {history.history['loss'][-1]}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]}")

# Make predictions on a few test samples
print("\nPredictions on first 5 test images:")
predictions = model.predict(X[5])
# Get the predicted class (index with highest probability)

print(predictions)








import matplotlib.pyplot as plt

# Select an image from the test set to predict
image_index = 5 # You can change this index to see different images

# Get the image and its actual label
image_to_predict = X[image_index].reshape(1, 784) # Reshape for the model

# Make a prediction using the trained model
prediction = model.predict(image_to_predict)

# Display the image
plt.imshow(prediction.reshape(28, 28), cmap='gray')

plt.axis('off') # Hide axes
plt.show()





