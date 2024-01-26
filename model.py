import numpy as np
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt


# >>> That's going to  be a classification which one is giving a predict crocodile or aligator.
# Our dataset include 1728 photos of aligators and 1979 photos of crocodiles

# Folder path
folder_path_aligator = "/Users/tonix/Desktop/Python/aligators vs crocodiles/alligator vs crocodile/alligator"
folder_path_crocodile =  "/Users/tonix/Desktop/Python/aligators vs crocodiles/alligator vs crocodile/crocodile"

# Files in path
aligator_list = os.listdir(folder_path_aligator)       # 1727
crocodile_list = os.listdir(folder_path_crocodile)     # 1978

# >>> Aligators
# Labels and photo of aligators
labels = []

num_images_aligator = len(aligator_list)
image_size = (250, 250)
channels = 3

# >>> Creating an array with photo of aligators and preprocessing
image_aligators = np.zeros((num_images_aligator, image_size[0], image_size[1], channels))
valid_indices_aligator = []

for index_aligator, file_name_aligator in enumerate(aligator_list):

    image_path_aligator = os.path.join(folder_path_aligator, file_name_aligator)

    with Image.open(image_path_aligator) as img_a:
        width_aligator, height_aligator = img_a.size

        if 125 < width_aligator < 370 and 125 < height_aligator < 370:

            if img_a != 'RGB':
                img_a = img_a.convert('RGB')

            img_a = img_a.resize((250, 250))
            image_aligator = np.array(img_a) / 255                    # Normalizing
            image_aligators[index_aligator] = image_aligator

            labels.append(0)
            valid_indices_aligator.append(index_aligator)

image_aligators = image_aligators[valid_indices_aligator]

# Same for crocodiles
# >>> Creating an array with photo of crocodiles and preprocessing
num_images_crocodiles = len(crocodile_list)

image_crocodiles = np.zeros((num_images_crocodiles, image_size[0], image_size[1], channels))
valid_indices_crocodile = []

for index_crocodile, file_name_crocodile in enumerate(crocodile_list):

    image_path_crocodile = os.path.join(folder_path_crocodile, file_name_crocodile)

    with Image.open(image_path_crocodile) as img_c:
        width_crocodile, height_crocodile = img_c.size

        if 125 < width_crocodile < 422 and 125 < height_crocodile < 422:

            if img_c != 'RGB':
                img_c = img_c.convert('RGB')

            img_c = img_c.resize((250, 250))
            image_crocodile = np.array(img_c) / 255                     # Normalizng
            image_crocodiles[index_crocodile] = image_crocodile

            labels.append(1)
            valid_indices_crocodile.append(index_crocodile)

image_crocodiles = image_crocodiles[valid_indices_crocodile]
labels = np.array(labels)   # Create an array

# After preprocessing we have 1836 crocodiles and 1409 aligators sum(3245)
# So we can split it on 2596 train photo and 649 test photo
shuffled_indices = np.random.RandomState(seed=2).permutation(len(labels))

images = (np.concatenate((image_aligators, image_crocodiles), axis=0)[shuffled_indices])
labels = labels[shuffled_indices]

x_train = images[: 2597]
y_train = labels[: 2597]

x_test = images[2596 :]
y_test = labels[2596 :]

# >>> Building a model
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(250, 250, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Dropout(rate=(0.25)),
    tf.keras.layers.Conv2D(48, (3, 3)),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Dropout(rate=0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense((192), activation='relu'),
    tf.keras.layers.Dropout(rate=0.15),
    tf.keras.layers.Dense((1), activation='sigmoid')

])

# >>> Compilation
model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])

# Model fit
history = model.fit(x_train, y_train, epochs=20)

# Model evaluate
model.evaluate(x_test, y_test, verbose=2)

