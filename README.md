# Speaker Recognition (Binary Classification)
This project implements a binary speaker recognition system using MFCC features and a classical machine-learning pipeline.

## Project Structure
```bash
Speaker_Detection/
│
├── src/
│   ├── main.py
│   ├── config/
│   ├── data/
│   ├── features/
│
├── data/
│   ├── target/         # target speaker .wav files
│   ├── other/          # non-target speaker .wav files
│   ├── metadata/       # auto-generated dataset_info.csv
│   └── processed/      # auto-generated .npy feature files
│
└── experiments/
    ├── models/
    └── logs/
```

## Setup
### code:
create env using python 3.12:
```bash
conda create -n speaker_recognition python=3.12
```
activate the env:
```bash
conda activate speaker_recognition
```
install dependencies:
```bash
pip install -r requirements.txt
```
### data:
Your audio data must be placed like this:
```bash
data/
  target/
      <audio files of the target speaker>.wav
  other/
      <audio files of all other speakers>.wav
```

## Usage
Run the entire pipeline:
```bash
python src/main.py
```
Clean all auto-generated files:
```bash
python src/main.py clean
```

## Pipeline Overview
### Metadata Preparation:
Extracts file paths + labels and creates:
```bash
data/metadata/dataset_info.csv
```
### Feature Extraction
MFCC features are extracted for each .wav file as:
```bash
X_train.npy
y_train.npy
X_test.npy
y_test.npy
```

## Credits
Developed by : Jewel.M.Jain  
Data collected by: Vaibhav  
Market Research by: Nikolas  
Course: Signal and Speech Processing  
Semester: 4th Year – Academia  