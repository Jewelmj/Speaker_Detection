import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_metadata(base_path="data"):
    target_dir = os.path.join(base_path, "target")
    other_dir = os.path.join(base_path, "other")
    metadata_path = os.path.join(base_path, "metadata")
    os.makedirs(metadata_path, exist_ok=True)

    data_rows = []

    for fname in os.listdir(target_dir):
        if fname.lower().endswith(".wav"):
            data_rows.append([os.path.join("data/target", fname), 1])

    for fname in os.listdir(other_dir):
        if fname.lower().endswith(".wav"):
            data_rows.append([os.path.join("data/other", fname), 0])

    df = pd.DataFrame(data_rows, columns=["filepath", "label"])

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_df["split"] = "train"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, test_df], ignore_index=True)

    save_path = os.path.join(metadata_path, "dataset_info.csv")
    final_df.to_csv(save_path, index=False)

    print(f"Saved metadata to {save_path}")
    print(final_df.head())