import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist

np.random.seed(5)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=500, validation_data=(X_test,y_test))

y_pred = np.argmax(model.predict(X_test), axis=-1)
print("Accuracy: %.2f" % metrics.accuracy_score(y_test,y_pred))

model.save('digitrecognizer_model.h5')


