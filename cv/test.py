import os
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

base_dir = os.path.join(os.getcwd(), 'split_dataset')

IMAGE_SIZE = 200
BATCH_SIZE = 64

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE)

model = tf.keras.models.load_model('../models/first_train')

true_labels = test_generator.classes
probabilities = model.predict_generator(test_generator)

y_pred = np.array([np.argmax(x) for x in probabilities])

print(confusion_matrix(true_labels, y_pred))

print(model.evaluate(tf.data.Dataset.from_generator(test_generator)))
