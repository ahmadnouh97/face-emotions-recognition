import tensorflow as tf
import sys
import os
import yaml

sys.path.append(os.path.abspath('./'))
from Code.config import Config
from Code.models import build_model
from Code.utils import read_data, plot_training_history

with open(Config.PARAMS_PATH) as f:
    params = yaml.load(f.read())['train']

x_train = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'x_train.pkl'))
x_val = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'x_val.pkl'))


y_train = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'y_train.pkl'))
y_val = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'y_val.pkl'))

model = build_model(
    class_num=int(params['class_num']),
    image_size=int(params['image_size']),
    learning_rate=float(params['learning_rate']),
    activation_func=str(params['activation_func']),
    dense_units=list(params['dense_units']),
    conv2d_units=list(params['conv2d_units']),
    conv2d_kernels=list(params['conv2d_kernels']),
    pool_sizes=list(params['pool_sizes'])
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x=x_train,
                    y=y_train,
                    validation_data=(x_val, y_val),
                    batch_size=int(params['batch_size']),
                    epochs=int(params['epochs']),
                    callbacks=[early_stopping])


plot_training_history(history, save_to=Config.PLOTS_DIR)

model.save(os.path.join(Config.MODEL_DIR, 'model.h5'))
