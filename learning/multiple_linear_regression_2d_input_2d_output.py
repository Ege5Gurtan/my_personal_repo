from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.callbacks import TensorBoard


# Define input and output data
x_train = np.array([[1,3], [2,4], [3,6], [4,9], [5,2]]) #[x1,x2]
y_train = np.array([[2,9], [4,12], [6,18], [8,27], [10,6]]) #[2x1,3x2]
x_test = np.array([[5,1], [6,2], [7,3], [8,1], [9,0]])
epochs = 10000
dimension_of_output_data = 2
dimension_of_input_data = 2
# Create the neural network model
model = Sequential()
model.add(Dense(dimension_of_output_data, input_dim = dimension_of_input_data))

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=epochs, verbose=0,callbacks=[tensorboard])

# Use the model to make predictions
y_pred = model.predict(x_test)

# Print the predicted values
print(y_pred)
