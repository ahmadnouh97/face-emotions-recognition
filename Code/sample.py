import tensorflow as tf
import sys
import os
import random
import yaml

sys.path.append(os.path.abspath('./'))
from Code.config import Config
from Code.utils import read_data, image_to_file

with open(Config.PARAMS_PATH) as f:
    params = yaml.load(f.read())['evaluate']

model = tf.keras.models.load_model(os.path.join(Config.MODEL_DIR, 'model.h5'))

x_test = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'x_test.pkl'))
y_test = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'y_test.pkl'))

data = list(zip(x_test, y_test))
samples = random.sample(data, 5)
x_samples = [x for x, _ in samples]
y_samples = [y for _, y in samples]

os.makedirs(Config.SAMPLES_DIR, exist_ok=True)

selected_emotions = list(params['selected_emotions'])
index_to_emotion = {i: label for i, label in enumerate(selected_emotions)}

predictions = model.predict(x_samples)
y_pred = [row.argmax() for row in predictions]
y_pred = [index_to_emotion.get(item) for item in list(y_pred)]
y_test = [index_to_emotion.get(item) for item in list(y_test)]

for i, (sample, prediction, label) in enumerate(zip(x_samples, y_pred, y_test)):
    image_to_file(sample, os.path.join(Config.SAMPLES_DIR, f'{i}.{prediction} ({label}).png'))
