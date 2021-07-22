import tensorflow as tf
import pandas as pd
import sys
import os
import yaml
import json
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath('./'))
from Code.config import Config
from Code.utils import read_data

with open(Config.PARAMS_PATH) as f:
    params = yaml.load(f.read())['evaluate']

selected_emotions = list(params['selected_emotions'])
index_to_emotion = {i: label for i, label in enumerate(selected_emotions)}

x_test = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'x_test.pkl'))
y_test = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'y_test.pkl'))

model = tf.keras.models.load_model(os.path.join(Config.MODEL_DIR, 'model.h5'))

predictions = model.predict(x_test)

y_pred = []
for row in predictions:
    y_pred.append(row.argmax())

y_test = [index_to_emotion.get(item) for item in list(y_test)]
y_pred = [index_to_emotion.get(item) for item in list(y_pred)]

class_report = classification_report(list(y_test), y_pred, output_dict=True)

metrics = {
    'Happy': class_report['Happy']['f1-score'],
    'Sad': class_report['Sad']['f1-score'],
    'Neutral': class_report['Neutral']['f1-score']
}

with open(os.path.join(Config.METRICS_DIR, 'metrics.json'), 'w') as file:
    json.dump(metrics, file)
