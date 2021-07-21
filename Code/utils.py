import os
import pickle as pkl
import matplotlib.pyplot as plt


def read_data(file_path):
    with open(file_path, 'rb') as file:
        data = pkl.load(file)
    return data


def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pkl.dump(data, file)


def plot_training_history(history, save_to):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig(os.path.join(save_to, 'accuracy.png'))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig(os.path.join(save_to, 'loss.png'))
