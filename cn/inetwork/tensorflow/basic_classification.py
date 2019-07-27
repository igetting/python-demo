import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.VERSION)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

save_path = 'training_0/mode'

model.save_weights(save_path)

loss, acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', acc)

predictions = model.predict(test_images)
# print(predictions[0])

print(np.argmax(predictions[0]))
print(test_labels[0])