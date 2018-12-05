# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
#from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10#1000#10#1#10
config.cnn_dropout = 0.2

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()
print(X_test.shape)

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_test.shape)

num_classes = y_train.shape[1]
print(num_classes)

# you may want to normalize the data here..
# normalize data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# create model
model=Sequential()
model.add(Reshape((28,28,1), input_shape=(28,28))) # change from 28x28 into 28x28x1
model.add(Dropout(config.cnn_dropout))
model.add(Conv2D(32, (3,3), padding='same', activation='relu')) #32
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss=config.loss, optimizer=config.optimizer,
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test), callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])

#model.fit(X_train[:100], y_train[:100], epochs=config.epochs, #validation_data=(X_test, y_test), 
#          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
print("Target", y_train[:2])
print("Predictions", model.predict(X_train[:2]))
