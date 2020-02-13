from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# print(train_images[0])


# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()
model.summary()

checkpoint_path1 = "training_1/cp.ckpt"
checkpoint_dir1 = os.path.dirname(checkpoint_path1)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path1,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # pass callback to trainingmodel = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(checkpoint_path1)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# include the epoch in the file name. (uses `str.format`)
checkpoint_path2 = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir2 = os.path.dirname(checkpoint_path2)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path2, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

latest = tf.train.latest_checkpoint(checkpoint_dir2)

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Save the weights
checkpoint_path3 = './training_3/my_checkpoint'
model.save_weights(checkpoint_path3)

# Restore the weights
model = create_model()
model.load_weights(checkpoint_path3)

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
checkpoint_path4 = './training_4/my_model.h5'
checkpoint_dir4 = os.path.dirname(checkpoint_path4)
if not os.path.exists(checkpoint_dir4):
    os.mkdir(checkpoint_dir4)
model.save(checkpoint_path4)

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model(checkpoint_path4)
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
