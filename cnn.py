import tensorflow as tf 
from tensorflow.keras  import layers,models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np 
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)
print(test_images.shape)

plt.imshow(train_images[0])
plt.title("First image in train_images")
plt.axis('off')
plt.show()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = to_categorical(train_labels, 10)
test_images = to_categorical(test_labels, 10)

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.MaxPooling2D(64, (3,3), activation = 'relu'))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")
test_pred = model.predict(test_images)
test_pred_labels = np.argmax(test_pred, axis=1)
test_labels_non_cat = np.argmax(test_labels, axis=1)

from sklearn.metrics import classification_report
cr = classification_report(test_labels_non_cat, test_pred_labels)
print(cr)