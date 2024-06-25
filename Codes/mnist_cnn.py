import keras
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#%%
plt.figure(figsize=(14,14))
x, y = 10, 4 
for i in range(40):  
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i], cmap='gray')
plt.show()
#%%
batch_size = 16 
num_classes = 10 
epochs = 3 

img_rows, img_cols = 28, 28


x_train = x_train.reshape(x_train.shape[0], 
                          img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], 
                        img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
#%%
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
#%%
# Model oluşturma

model = Sequential() 
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = tensorflow.keras.optimizers.Adam(),
             metrics = ['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#%%
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(dpi=400)
plt.plot(epochs, train_accuracy, label='Eğitim Doğruluğu')
plt.plot(epochs, val_accuracy, label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()   

# Kayıp grafiği
plt.figure(dpi=400)
plt.plot(epochs, train_loss, label='Eğitim Kaybı')
plt.plot(epochs, val_loss, label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


# model.save('mnist_model.h5')