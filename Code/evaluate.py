import tensorflow as tf
import sys
import os
import yaml
import json
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

sys.path.append(os.path.abspath('./'))
from Code.config import Config
from Code.models import build_model
from Code.utils import save_to_csv

with open(Config.PARAMS_PATH) as f:
    params = yaml.load(f.read())['train']

classes = list(params['classes'])
index_to_class = {i: label for i, label in enumerate(classes)}

image_size = int(params['image_size'])
classes = list(params['classes'])
input_shape = (None, image_size, image_size, 1)
# x_test = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'x_test.pkl'))
# y_test = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'y_test.pkl'))

test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=int(params['rotation_range']),
    width_shift_range=float(params['width_shift_range']),
    height_shift_range=float(params['height_shift_range']),
    brightness_range=list(params['brightness_range']),
    zoom_range=float(params['zoom_range']),
    horizontal_flip=bool(params['horizontal_flip']),
    vertical_flip=bool(params['vertical_flip']),
)

test_generator = test_data_generator.flow_from_directory(
    Config.TEST_DATA_DIR,
    target_size=(image_size, image_size),
    classes=classes,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

model = build_model(
    class_num=int(params['class_num']),
    image_size=int(params['image_size']),
    learning_rate=float(params['learning_rate']),
    activation_func=str(params['activation_func']),
    dense_units=list(params['dense_units']),
    conv2d_units=list(params['conv2d_units']),
    conv2d_kernels=list(params['conv2d_kernels']),
    pool_sizes=list(params['pool_sizes']),
    dropout=float(params['dropout'])
)

model.build(input_shape)
model.load_weights(os.path.join(Config.MODEL_DIR, 'model.tf'))

predictions = model.predict(
    test_generator
)

y_pred = [row.argmax() for row in predictions]

y_test = [index_to_class.get(item) for item in test_generator.labels.tolist()]
y_pred = [index_to_class.get(item) for item in list(y_pred)]

predictions_dict = {
    'actual': y_test,
    'predicted': y_pred
}

class_report = classification_report(y_test, y_pred, output_dict=True)

print(class_report)

happy_f1_metrics = {
    'f1-score': class_report['Happy']['f1-score']
}

happy_precision_metrics = {
    'precision': class_report['Happy']['precision']
}

happy_recall_metrics = {
    'recall': class_report['Happy']['recall']
}

sad_f1_metrics = {
    'f1-score': class_report['Sad']['f1-score']
}

sad_precision_metrics = {
    'precision': class_report['Sad']['precision']
}

sad_recall_metrics = {
    'recall': class_report['Sad']['recall']
}

neutral_f1_metrics = {
    'f1-score': class_report['Neutral']['f1-score']
}

neutral_precision_metrics = {
    'precision': class_report['Neutral']['precision']
}

neutral_recall_metrics = {
    'recall': class_report['Neutral']['recall']
}

total_f1_metrics = {
    'f1-score': f1_score(y_test, y_pred, average='macro')
}

total_precision_metrics = {
    'f1-score': precision_score(y_test, y_pred, average='macro')
}

total_recall_metrics = {
    'f1-score': recall_score(y_test, y_pred, average='macro')
}

# f1_metrics = {
#     'Happy': class_report['Happy']['f1-score'],
#     'Sad': class_report['Sad']['f1-score'],
#     'Neutral': class_report['Neutral']['f1-score']
# }

# precision_metrics = {
#     'Happy': class_report['Happy']['precision'],
#     'Sad': class_report['Sad']['precision'],
#     'Neutral': class_report['Neutral']['precision']
# }

# recall_metrics = {
#     'Happy': class_report['Happy']['recall'],
#     'Sad': class_report['Sad']['recall'],
#     'Neutral': class_report['Neutral']['recall']
# }

os.makedirs(Config.METRICS_DIR, exist_ok=True)

with open(os.path.join(Config.METRICS_DIR, 'f1_score.json'), 'w') as file:
    json.dump(total_f1_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'precision.json'), 'w') as file:
    json.dump(total_precision_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'recall.json'), 'w') as file:
    json.dump(total_recall_metrics, file)

# ---------------------------------------------------------------------------

with open(os.path.join(Config.METRICS_DIR, 'happy_f1_score.json'), 'w') as file:
    json.dump(happy_f1_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'happy_precision.json'), 'w') as file:
    json.dump(happy_precision_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'happy_recall.json'), 'w') as file:
    json.dump(happy_recall_metrics, file)

# ---------------------------------------------------------------------------
with open(os.path.join(Config.METRICS_DIR, 'sad_f1_score.json'), 'w') as file:
    json.dump(sad_f1_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'sad_precision.json'), 'w') as file:
    json.dump(sad_precision_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'sad_recall.json'), 'w') as file:
    json.dump(sad_recall_metrics, file)

# ---------------------------------------------------------------------------
with open(os.path.join(Config.METRICS_DIR, 'neutral_f1_score.json'), 'w') as file:
    json.dump(neutral_f1_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'neutral_precision.json'), 'w') as file:
    json.dump(neutral_precision_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'neutral_recall.json'), 'w') as file:
    json.dump(neutral_recall_metrics, file)

# ---------------------------------------------------------------------------

save_to_csv(predictions_dict, os.path.join(Config.PLOTS_DIR, 'classes.csv'))
