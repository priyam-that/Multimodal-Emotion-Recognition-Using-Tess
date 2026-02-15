# Multimodal Emotion Recognition (TESS)

This repository contains the implementation of a Multimodal Emotion Recognition system using the Toronto Emotional Speech Set (TESS). The system leverages Speech (Audio) and Text (Transcripts) modalities.

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

2.  **Individual Training**:
    To retrain any pipeline:
    ```bash
    python3 project/speech_pipeline/train.py
    # python3 project/text_pipeline/train.py
    # python3 project/fusion_pipeline/train.py
    ```
## A. Architecture Decisions

### 1. Preprocessing
- **Speech**: Audio samples were resampled to 16kHz and silence-trimmed to remove non-informative segments. We used a fixed length of 200 time steps (padding/truncating) to enable batch processing.
- **Text**: Transcripts were cleaned and tokenized using Keras Tokenizer, with sequences padded to 20 tokens.

### 2. Feature Extraction
- **Speech**: We extracted **40 MFCCs** (Mel-frequency cepstral coefficients) per time step. MFCCs are the standard for speech processing as they approximate the human auditory system's response.
- **Text**: An **Embedding layer** (dim=50) was used to transform integer tokens into dense vectors, capturing semantic relationships (though limited in TESS).

### 3. Temporal/Contextual Modelling
- **Speech**: A **Bidirectional LSTM (Bi-LSTM)** network (2 layers: 64 $\to$ 32 units) was chosen. Bi-LSTMs capture temporal dynamics in both forward and backward directions, essential for speech emotion recognition where intonation evolves over time.
- **Text**: A single-layer **Bi-LSTM** (32 units) was used to capture contextual dependencies between words in the transcript.

### 4. Fusion
- **Method**: **Late Fusion**. We extracted the high-level learned features (outputs of dense layers) from the pre-trained Speech and Text models and concatenated them.
- **Rationale**: Late fusion allows each modality to learn its optimal representation independently before combination. This is robust when one modality (Speech) is significantly more informative than the other (Text), preventing early noise interference.

### 5. Classifier
- A fully connected **Dense Layer** with Softmax activation maps the fused representation to the 7 emotion classes.

---

## B. Experiments

We evaluated three model variants on a 20% held-out test set.

| Model Pipeline | Accuracy | Macro F1-Score | Computational Cost |
| :--- | :--- | :--- | :--- |
| **Speech-only** | **100.00%** | **1.00** | Medium (Bi-LSTM) |
| **Text-only** | **14.29%** | **0.04** | Low |
| **Multimodal** | **100.00%** | **1.00** | High (Combined) |

**Comparative Analysis**:
- The **Speech model** achieved perfect accuracy, confirming that TESS's acoustic features are highly discriminative.
- The **Text model** performed at random chance (~14%) because the transcripts (e.g., "Say the word date") are identical across different emotion labels.
- The **Multimodal Fusion model** successfully matched the Speech performance, demonstrating that the fusion mechanism effectively selected the informative speech features while ignoring the uninformative text features.

---

## C. Analysis

### 1. Classification Difficulty
- **Easiest/Hardest**: All 7 emotions (Anger, Disgust, Fear, Happiness, Neutral, Pleasant Surprise, Sadness) were classified with 100% accuracy. The high quality and exaggerated nature of the acted TESS dataset made separation extremely effective for deep learning models.

### 2. Role of Fusion
Fusion provides redundancy. In this specific case, it did not boost accuracy over Speech-only because Speech was already perfect. However, the system demonstrated robustness: adding a non-informative modality (Text) did **not degrade** performance, proving the efficacy of the late fusion architecture.

### 3. Error Analysis
- **Speech/Fusion**: No errors (0 misclassifications).
- **Text**: Fails universally due to ambiguity.
    - *Case 1*: Input "Say the word youth" (True: Sad) $\to$ Predicted: Neutral.
    - *Case 2*: Input "Say the word youth" (True: Happy) $\to$ Predicted: Neutral.
    - *Reason*: The inputs are identical strings; the model cannot distinguish them.

### 4. Learned Representations (t-SNE)
Visualizations of the learned feature spaces (see `Results/plots/`):
- **Speech/Fusion**: Show 7 distinct, well-separated clusters, validating the model's ability to disentangle emotion.
- **Text**: Shows a single overlapping cluster, confirming lack of separability.

