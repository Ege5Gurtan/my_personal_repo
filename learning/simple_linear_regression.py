from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.callbacks import TensorBoard


# Define input and output data
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10]) #[2x]
x_test = np.array([5, 6, 7, 8, 9])
epochs = 10000

# Create the neural network model
model = Sequential()
model.add(Dense(1, input_dim = 1))


tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=epochs, verbose=0,callbacks=[tensorboard])

# Use the model to make predictions
y_pred = model.predict(x_test)

# Print the predicted values
print(y_pred)
