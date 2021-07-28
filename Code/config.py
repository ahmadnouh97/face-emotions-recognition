import os


class Config:
    ROOT = os.path.join('./')
    PARAMS_PATH = os.path.join(ROOT, 'params.yaml')
    DATA_DIR = os.path.join(ROOT, 'Data')
    PREPARED_DATA_DIR = os.path.join(DATA_DIR, 'prepared_data')
    TRAIN_DATA_DIR = os.path.join(PREPARED_DATA_DIR, 'train')
    VAL_DATA_DIR = os.path.join(PREPARED_DATA_DIR, 'val')
    TEST_DATA_DIR = os.path.join(PREPARED_DATA_DIR, 'test')

    PLOTS_DIR = os.path.join(ROOT, 'Plots')
    MODEL_DIR = os.path.join(ROOT, 'Model')
    METRICS_DIR = os.path.join(ROOT, 'Metrics')
    SAMPLES_DIR = os.path.join(ROOT, 'Samples')
