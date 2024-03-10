import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'Data_sets/ccn_datasets/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    'Data_sets/ccn_datasets/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=2, activation='softmax'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=validation_generator, epochs=10)


import numpy as np
from keras.preprocessing import image


test_image = image.load_img('Data_sets/ccn_datasets/single_prediction/cat_or_dog_1.jpg',
                            target_size=(64, 64))
# Convert the image to a numpy array and rescale it
test_image = image.img_to_array(test_image) / 255.0
# Add an extra dimension to the image (to represent the batch size)
test_image = np.expand_dims(test_image, axis=0)
# Predict the class of the image
result = cnn.predict(test_image)
# Get the index of the class with the highest probability
predicted_class = np.argmax(result, axis=1)
# Assuming class_indices are sorted alphabetically
class_labels = list(training_set.class_indices.keys())
prediction = class_labels[predicted_class[0]]

print(prediction)