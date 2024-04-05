from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential

def create_model(training_set):
    # Define the model
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=12, 
                    input_shape=(train_features.shape[1], 4)), # Apply 1D convolution with 32 output channels and a kernel size of 12
            MaxPooling1D(pool_size=4), # Add a max pooling layer to reduce the output dimension of the Conv1D layer
            Flatten(), # Flatten the output of the max pooling layer to be consumable by the dense layer
            Dense(16, activation='relu'), # Add two dense layers to compress features to 16 dimensions and then to 2 dimensions for binary classification
            Dense(2, activation='softmax'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                metrics=['binary_accuracy'])

    return model
