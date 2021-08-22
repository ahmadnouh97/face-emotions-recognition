import tensorflow as tf
import sys
import os
import yaml

sys.path.append(os.path.abspath('./'))
from Code.config import Config
from Code.models import build_model
from Code.utils import plot_training_history, save_to_csv

with open(Config.PARAMS_PATH) as f:
    params = yaml.load(f.read())['train']

image_size = int(params['image_size'])
batch_size = int(params['batch_size'])
classes = list(params['classes'])

train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=int(params['rotation_range']),
    width_shift_range=float(params['width_shift_range']),
    height_shift_range=float(params['height_shift_range']),
    brightness_range=list(params['brightness_range']),
    zoom_range=float(params['zoom_range']),
    horizontal_flip=bool(params['horizontal_flip']),
    vertical_flip=bool(params['vertical_flip'])
)

train_generator = train_data_generator.flow_from_directory(
    Config.TRAIN_DATA_DIR,
    target_size=(image_size, image_size),
    color_mode='grayscale',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=int(params['rotation_range']),
    width_shift_range=float(params['width_shift_range']),
    height_shift_range=float(params['height_shift_range']),
    brightness_range=list(params['brightness_range']),
    zoom_range=float(params['zoom_range']),
    horizontal_flip=bool(params['horizontal_flip']),
    vertical_flip=bool(params['vertical_flip']),
)

val_generator = val_data_generator.flow_from_directory(
    Config.VAL_DATA_DIR,
    target_size=(image_size, image_size),
    color_mode='grayscale',
    classes=classes,
    class_mode='categorical',
    shuffle=False
)

# x_train = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'x_train.pkl'))
# x_val = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'x_val.pkl'))
# y_train = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'y_train.pkl'))
# y_val = read_data(os.path.join(Config.PREPARED_DATA_DIR, 'y_val.pkl'))

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

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001)

steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = val_generator.n // val_generator.batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=int(params['epochs']),
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)

# history = model.fit(x=x_train,
#                     y=y_train,
#                     validation_data=(x_val, y_val),
#                     batch_size=batch_size,
#                     epochs=int(params['epochs']),
#                     callbacks=[early_stopping])


plot_training_history(history, save_to=Config.PLOTS_DIR)

history_val_loss = {
    'epoch': [epoch for epoch in range(len(history.history.get('val_loss')))],
    'val_loss': [loss for loss in history.history.get('val_loss')]
}
history_val_accuracy = {
    'epoch': [epoch for epoch in range(len(history.history.get('val_loss')))],
    'val_accuracy': [accuracy for accuracy in history.history.get('val_accuracy')]
}

save_to_csv(history_val_loss, os.path.join(Config.PLOTS_DIR, 'val_loss.csv'))
save_to_csv(history_val_accuracy, os.path.join(Config.PLOTS_DIR, 'val_accuracy.csv'))

model.save_weights(os.path.join(Config.MODEL_DIR, 'model.tf'), save_format='tf')
