import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
import cv2

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display sample MNIST images
unique, counts = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

indexes = np.random.randint(0, x_train.shape[0], 25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5, 5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.show()

# Preprocess MNIST data
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build and train MNIST model
model_mnist = Sequential()
model_mnist.add(Dense(512, input_dim=784))
model_mnist.add(Activation('relu'))
model_mnist.add(Dropout(0.2))
model_mnist.add(Dense(512))
model_mnist.add(Activation('relu'))
model_mnist.add(Dropout(0.2))
model_mnist.add(Dense(10))
model_mnist.add(Activation('softmax'))

model_mnist.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_mnist.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

print(model_mnist.evaluate(x_test, y_test))

model_mnist.save('mnist_model.h5')

# Load and preprocess a sample image for MNIST model prediction
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image.reshape(1, 28 * 28) / 255.0
prediction = model_mnist.predict(image)

print(np.argmax(prediction))

# Load and preprocess MNIST data for CNN model
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure(figsize=(14, 14))
x, y = 10, 4
for i in range(40):
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i], cmap='gray')
plt.show()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_test = tf.keras.utils.to_categorical(y_test, 10)
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Build and train MNIST CNN model
model_mnist_cnn = Sequential()
model_mnist_cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_mnist_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_mnist_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_mnist_cnn.add(Flatten())
model_mnist_cnn.add(Dense(256, activation='relu'))
model_mnist_cnn.add(Dropout(0.2))
model_mnist_cnn.add(Dense(10, activation='softmax'))

model_mnist_cnn.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

hist = model_mnist_cnn.fit(x_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(x_test, y_test))

train_accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(hist.history['accuracy']) + 1)

plt.figure(dpi=400)
plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

input_shape = train_images.shape[1:]
num_classes = len(set(train_labels.flatten()))

print(input_shape)
print(num_classes)
print(train_images.shape)

# Normalize CIFAR-10 data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build and train CIFAR-10 CNN model
model_cifar10 = Sequential()
model_cifar10.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model_cifar10.add(MaxPooling2D((2, 2)))
model_cifar10.add(Conv2D(64, (3, 3), activation='relu'))
model_cifar10.add(MaxPooling2D((2, 2)))
model_cifar10.add(Flatten())
model_cifar10.add(Dense(128, activation='relu'))
model_cifar10.add(Dense(num_classes, activation='softmax'))

model_cifar10.summary()

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_cifar10.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model_cifar10.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
