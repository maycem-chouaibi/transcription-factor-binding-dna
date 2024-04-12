from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential

def create_model(input):
  
    return Sequential([
    Conv1D(filters=32, kernel_size=12, input_shape=input_shape),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
    ])