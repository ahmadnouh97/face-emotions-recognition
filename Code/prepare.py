import yaml
import sys
import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm

sys.path.append(os.path.abspath('./'))
from Code.config import Config
from Code.utils import save_data, image_to_file

os.makedirs(Config.PREPARED_DATA_DIR, exist_ok=True)
os.makedirs(Config.TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(Config.VAL_DATA_DIR, exist_ok=True)
os.makedirs(Config.TEST_DATA_DIR, exist_ok=True)

params = yaml.safe_load(open(Config.PARAMS_PATH))['prepare']
data_path = os.path.join(Config.DATA_DIR, 'fer2013.csv')

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_to_index = {label: i for i, label in enumerate(emotion_labels)}
raw_data = pd.read_csv(data_path, index_col=False)

image_size = int(sqrt(len(raw_data.pixels[0].split())))

print(f'image dimensions: {image_size} x {image_size}')

selected_emotions = list(params['selected_emotions'])
emotion_to_index = {label: i for i, label in enumerate(selected_emotions)}

data = raw_data[
    raw_data['emotion'].apply(lambda emotion_idx: emotion_labels[emotion_idx] in selected_emotions)].reset_index(
    drop=True)

for emotion in selected_emotions:
    data['emotion'] = data['emotion'].replace(to_replace=label_to_index.get(emotion), value=emotion)


def str_to_image(pixels: str):
    return np.mat(pixels).reshape(image_size, image_size)


y = np.array(data['emotion'].apply(emotion_to_index.get))
x = np.array(list(data['pixels'].apply(str_to_image)))
x = x[:, :, :, np.newaxis]

data_split_dict = data['Usage'].value_counts().to_dict()
t = data_split_dict.get('Training')
v = data_split_dict.get('PublicTest')


def save_image_to_dir(image, label, directory, name):
    dir_path = os.path.join(directory, label)
    os.makedirs(dir_path, exist_ok=True)
    image_to_file(x_sample, file_path=os.path.join(dir_path, name))


train_i = 0
val_i = 0
test_i = 0

for i, (x_sample, y_sample) in tqdm(enumerate(zip(x, y))):
    if i < t:
        save_image_to_dir(x_sample, selected_emotions[y_sample], Config.TRAIN_DATA_DIR, f'train_{train_i}.png')
        train_i += 1
    elif t <= i < t + v:
        save_image_to_dir(x_sample, selected_emotions[y_sample], Config.VAL_DATA_DIR, f'val_{val_i}.png')
        val_i += 1
    else:
        save_image_to_dir(x_sample, selected_emotions[y_sample], Config.TEST_DATA_DIR, f'test_{test_i}.png')
        test_i += 1

# x_train = x[:t]
# x_val = x[t:t + v]
# x_test = x[t + v:]
#
# y_train = y[:t]
# y_val = y[t:t + v]
# y_test = y[t + v:]
#
# save_data(x_train, os.path.join(Config.PREPARED_DATA_DIR, 'x_train.pkl'))
# save_data(x_val, os.path.join(Config.PREPARED_DATA_DIR, 'x_val.pkl'))
# save_data(x_test, os.path.join(Config.PREPARED_DATA_DIR, 'x_test.pkl'))
#
# save_data(y_train, os.path.join(Config.PREPARED_DATA_DIR, 'y_train.pkl'))
# save_data(y_val, os.path.join(Config.PREPARED_DATA_DIR, 'y_val.pkl'))
# save_data(y_test, os.path.join(Config.PREPARED_DATA_DIR, 'y_test.pkl'))
