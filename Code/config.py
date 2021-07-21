import os


class Config:
    ROOT = os.path.join('./')
    PARAMS_PATH = os.path.join(ROOT, 'params.yaml')
    DATA_DIR = os.path.join(ROOT, 'Data')
    PREPARED_DATA_DIR = os.path.join(DATA_DIR, 'prepared_data')
    PLOTS_DIR = os.path.join(ROOT, 'Plots')
    MODEL_DIR = os.path.join(ROOT, 'Model')
