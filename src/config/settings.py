# main settings
METADATA_FILE = "data/metadata/dataset_info.csv"
METADATA_DIR = "data/metadata"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "experiments/models"
EVAL_DIR = "experiments/evaluation"
LOG_DIR = "experiments/log"

# data preperation settings
DATASET_METADATA_PATH = "data/metadata/dataset_info.csv"

# feature extraction settings
FEATURES_TO_USE = ["mfcc"]   # ["mfcc", "delta", "deltadelta", "spectral", "melspec"]
FEATURE_TYPES = ["mfcc", "delta", "melspec"]

# training settings
MODEL_TO_USE = "svm" # ["svm", "logistic", "random_forest"]

# inference settings
INFERENCE_MODEL = "svm"