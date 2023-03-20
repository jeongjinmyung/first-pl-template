sweep_config = {
    "method": "random",   # Random search
    'name': 'sweep',
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        'batch_size': {'values': [16, 32, 64]},
        "n_layer_1": {
            # Choose from pre-defined values
            "values": [64, 128, 256]
        },
        "n_layer_2": {
            # Choose from pre-defined values
            "values": [64, 128, 256]
        },
        "lr": {
            # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -9.21,   # exp(-9.21) = 1e-4
            "max": -4.61    # exp(-4.61) = 1e-2
        }
    }
}

# training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 3
MIN_EPOCHS = 1
MAX_EPOCHS = 3
N_LAYER_1 = 128
N_LAYER_2 = 256
SEED = 42

# dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16
