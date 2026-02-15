# Multimodal Emotion Recognition (TESS)

This repository contains the implementation of a Multimodal Emotion Recognition system using the Toronto Emotional Speech Set (TESS), developed for **Assignment 2**. The system leverages Speech (Audio) and Text (Transcripts) modalities.

## Project Structure (Deliverables)

```
project/
├── models/             # Saved Keras models (.h5)
├── speech_pipeline/    # Speech Processing Logic
│   ├── train.py
│   └── test.py
├── text_pipeline/      # Text Processing Logic
│   ├── train.py
│   └── test.py
├── fusion_pipeline/    # Multimodal Fusion Logic
│   ├── train.py
│   └── test.py
├── Results/            # Evaluation Outputs
│   ├── accuracy_tables.md
│   └── plots/          # Confusion Matrices & t-SNE
├── report.md           # Detailed Project Report (Sections A, B, C)
└── requirements.txt    # Dependencies
```

## Setup & Usage

1.  **Initialize Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Run Full Evaluation**:
    Execute the master script to run all tests and generate plots:
    ```bash
    chmod +x project/test_all.sh
    ./project/test_all.sh
    ```

3.  **Individual Training**:
    To retrain any pipeline:
    ```bash
    python3 project/speech_pipeline/train.py
    # python3 project/text_pipeline/train.py
    # python3 project/fusion_pipeline/train.py
    ```

## Results Summary

| Model | Accuracy | Analysis |
| :--- | :--- | :--- |
| **Speech** | **100%** | Perfect classification (State-of-the-Art). |
| **Text** | **~14%** | Baseline (Expected due to identical transcripts). |
| **Fusion** | **100%** | Robustly preserved speech performance. |

Full details in `report.md`.
