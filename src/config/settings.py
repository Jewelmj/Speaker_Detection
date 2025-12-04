# main settings
METADATA_FILE = "data/metadata/dataset_info.csv"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "experiments/models"
EVAL_DIR = "experiments/evaluation"

# data preperation settings
DATASET_METADATA_PATH = "data/metadata/dataset_info.csv"

# feature extraction settings
FEATURES_TO_USE = ["mfcc"]   # ["mfcc", "delta"]

# training settings
MODEL_TO_USE = "svm" # ["svm", "logistic", "random_forest"]

# inference settings
INFERENCE_MODEL = "svm"