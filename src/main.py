import os
import sys
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data.mertadata_prep import prepare_metadata
from src.features.audio_feature_extractor import batch_extract_features
from src.models.trainer import train_selected_model
from src.models.infer import run_inference

from src.config.settings import METADATA_FILE, PROCESSED_DIR, MODEL_DIR, EVAL_DIR, METADATA_DIR
def metadata_exists():
    return os.path.exists(METADATA_FILE)

def features_exist():
    required = [
        "X_train.npy", "y_train.npy",
        "X_test.npy", "y_test.npy",
    ]
    return all(os.path.exists(os.path.join(PROCESSED_DIR, f)) for f in required)

def model_exists():
    if not os.path.exists(MODEL_DIR):
        return False
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".joblib") or f.endswith(".pkl"):
            return True
    return False

def clean_all():
    print("Cleaning metadata, processed features, and models...")

    folders = [
        METADATA_DIR,
        PROCESSED_DIR,
        MODEL_DIR,
        EVAL_DIR,
    ]

    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

        gitkeep_path = os.path.join(folder, ".gitkeep")
        with open(gitkeep_path, "w") as f:
            f.write("")

    print("Clean completed.")

def load_data():
    print("Running Speaker Recognition Pipeline...\n")

    if not metadata_exists():
        print("Metadata missing → generating dataset_info.csv...")
        prepare_metadata()
    else:
        print("Metadata found → skipping.")

    if not features_exist():
        print("Extracted features missing → extracting MFCC features...")
        batch_extract_features()
    else:
        print("Features found → skipping.")

    print("\nPipeline completed successfully!")

def main():
    load_data()

    if len(sys.argv) == 1:
        print("\nNo command provided!")
        print("Use one of the following commands:")
        print("  python src/main.py clean")
        print("  python src/main.py train")
        print("  python src/main.py infer path/to/audio.wav")
        return
    elif sys.argv[1] == "clean":
        clean_all()
    elif sys.argv[1] == "train":
        print("\nTraining model...")
        train_selected_model()

    elif sys.argv[1] == "infer":
        if len(sys.argv) < 3:
            print("Missing audio file.\nUsage:")
            print("python src/main.py infer path/to/audio.wav")

        audio_path = sys.argv[2]

        print(f"\nRunning inference on: {audio_path}")
        run_inference(audio_path)
    
if __name__ == "__main__":
    main()