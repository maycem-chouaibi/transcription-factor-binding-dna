import data_handling as dh
import model as m
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input

train_features, test_features, train_labels, test_labels = dh.split_data(*dh.one_hot_encode(*dh.load_preprocess_data()))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential

# Define the input shape
input_shape = (train_features.shape[1], 4)

# Define the model
model = m.create_model(input_shape)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Train the model
history = model.fit(train_features, train_labels, epochs=50, verbose=0, validation_split=0.25)

# Evaluate the model
loss, accuracy = model.evaluate(test_features, test_labels)

sequence_index = 37

# Define input tensor
x_value = tf.convert_to_tensor(test_features[sequence_index][np.newaxis, ...], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x_value)
    predictions = model(x_value)
    saliency = tape.gradient(predictions[:, 1], x_value)  # Compute gradients with respect to the output of the second class (assuming binary classification)

gradients = saliency.numpy()
sal = np.abs(gradients[0])  # Take absolute values of gradients

# Plot saliency map
plt.figure(figsize=[16,5])
plt.plot(np.arange(len(sal)), sal, color='C1')
plt.xlabel('Base Position')
plt.ylabel('Magnitude of Saliency Values')
plt.title('Saliency Map for Bases in Sequence')
plt.show()
