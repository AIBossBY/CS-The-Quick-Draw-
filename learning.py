import numpy as np
import tensorflow as tf

#load dataset
x_train, y_train, x_test  = np.load('dataset_participant.npz').values()
dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
n=len(dataset)

#shuffle and divide
dataset.shuffle(n)
train_dataset = dataset.take(int(n*0.8)).batch(1000)
test_dataset = dataset.skip(int(n*0.8)).batch(1000)

train_dataset.prefetch(2)

#model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(100)
])

#loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='Adam', loss=loss_fn, metrics=['accuracy'])

#train
model.fit(train_dataset, epochs=50)

#evaluate
model.evaluate(test_dataset, verbose=2)