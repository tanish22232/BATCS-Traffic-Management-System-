import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split # type: ignore

# Load annotations
df = pd.read_csv("frameAnnotationsBOX.csv", sep=";")
df = df[df["Annotation tag"].isin(["stop", "go", "warning", "goLeft", "stopLeft", "warningLeft"])]

# Map labels to standard classes
label_map = {
    "stop": "red",
    "go": "green",
    "warning": "yellow",
    "goLeft": "green_left",
    "stopLeft": "red_left",
    "warningLeft": "yellow_left"
}
df["label"] = df["Annotation tag"].map(label_map)

# Create folders
os.makedirs("cropped_dataset/train", exist_ok=True)
os.makedirs("cropped_dataset/val", exist_ok=True)

# Split into train/val (80/20)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Crop and save images
def crop_and_save(row, split):
    img = Image.open(row["Filename"])
    crop = img.crop((row["x1"], row["y1"], row["x2"], row["y2"]))
    save_dir = f"cropped_dataset/{split}/{row['label']}"
    os.makedirs(save_dir, exist_ok=True)
    crop.save(f"{save_dir}/{os.path.basename(row['Filename'])}")

train_df.apply(lambda x: crop_and_save(x, "train"), axis=1)
val_df.apply(lambda x: crop_and_save(x, "val"), axis=1)