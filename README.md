<img width="2722" height="1568" alt="SemEvals" src="https://github.com/user-attachments/assets/7718d831-1992-4581-a9d7-3e72a99ca0a1" />

# SemEval-2026 Task 2: Temporal Mixture-of-Experts for Longitudinal Valence and Arousal Prediction

**Team Name:** CITD@UIT

This repository contains the official system description and implementation for our participation in **SemEval-2026 Task 2**. We propose a unified framework leveraging [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) augmented with a **Temporal Mixture-of-Experts (MoE)** architecture to capture longitudinal affect shifts.

## 🏆 Competition Achievements
Our team (**CITD@UIT**) achieved competitive results on the official SemEval-2026 Task 2 leaderboard:
* **Subtask 1 (Longitudinal Affect Assessment):** Rank **9/26** (Top 10).
* **Subtask 2A (State Change Forecasting):** Rank **5/15** (Top 5).

## 🚀 Model Weights
You can access our fine-tuned models on Hugging Face:
👉 [TheSon2202/Temporal-MoEs-RoBERTa](https://huggingface.co/TheSon2202/Temporal-MoEs-RoBERTa)

You can try the demo here (Gradio):
👉 **[Temporal-MoEs-RoBERTa Emotional State-Change Forecaster](https://huggingface.co/spaces/TheSon2202/temporal-moes-roberta-sentiment)**

## 🇻🇳 Vietnamese Sentiment Analysis (BamiBERT-MoE)
We have also developed a high-performance **Mixture-of-Experts (MoE)** architecture for Vietnamese sentiment analysis, built upon the **BamiBERT** backbone.
👉 **[TheSon2202/bamibert-moe-sentiment](https://huggingface.co/TheSon2202/bamibert-moe-sentiment)**

*   **Key Features:** Integrated custom MoE layers utilizing **Sparse Dynamic Expert Routing** to balance representation capacity with inference efficiency.
*   **Performance:** Achieved an **F1-Score (Macro) of 0.9068** on the benchmark public test set.
*   **Live Demo:** 👉 **[Try the Vietnamese Sentiment Demo here](https://huggingface.co/spaces/TheSon2202/bamibert-moe-vietnamese-sentiment-analytst)**

## System Description

Our system employs a unified framework based on `cardiffnlp/twitter-roberta-base-sentiment-latest`, augmented with a Temporal Mixture-of-Experts (MoE) head to capture longitudinal emotional shifts.

### Training Strategy

We optimized the models using the Concordance Correlation Coefficient (CCC) Loss to maximize the correlation between predictions and ground truth.

* **Subtask 1 Loss:** `0.4 * Loss_Valence + 0.6 * Loss_Arousal`
    * Learning Rate: `2e-5` | Max Seq Length: `384`

* **Subtask 2A Loss:** `0.5 * Loss_Valence + 0.5 * Loss_Arousal`
    * Learning Rate: `2e-5` | Max Seq Length: `512`

* **Subtask 2B Loss:** `0.5 * Loss_Valence + 0.5 * Loss_Arousal`
    * Learning Rate: `3e-5` | Max Seq Length: `512`

### Examples

#### Subtask 1
* **Training Instance:**
    * **Input:** `"I hate mondays </s> Traffic is terrible </s> Late for work </s> But at least I have coffee"`
    * **Target Output:** `Valence: 0.65, Arousal: 0.40`
* **Inference Instance:**
    * **Input:** `"So tired </s> Want to sleep </s> Almost done </s> Finally going home!"`
    * **Predicted Output:** `Valence: 0.72, Arousal: 0.35`

#### Subtask 2A 
* **Training Instance:**
    * **Input (Text):** `"Having a great day </s> Lunch was amazing </s> ... </s> Feeling relaxed"` (Window $k=8$)
    * **Input (Tabular):** `Current_Valence: 0.8, Current_Arousal: 0.2`
    * **Target Output:** `Change_Valence: -0.10, Change_Arousal: -0.05`
* **Inference Instance:**
    * **Input (Text):** `"Missed the bus </s> Late again </s> ... </s> Ready to work"`
    * **Input (Tabular):** `Current_Valence: 0.3, Current_Arousal: 0.7`
    * **Predicted Output:** `Change_Valence: +0.25, Change_Arousal: -0.30`

#### Subtask 2B 
* **Training Instance:**
    * **Input (Text):** `<s> I failed my exam </s> Feeling hopeless </s> </s> I passed the re-test! </s> So happy now </s>`
    * **Input (Tabular):** `Mean_Valence_H1: 0.20, Mean_Arousal_H1: 0.50`
    * **Target Output:** `Change_Valence: +0.60, Change_Arousal: +0.10`
* **Inference Instance:**
    * **Input (Text):** `<s> Just normal day </s> Nothing new </s> </s> Got a promotion </s> Excited! </s>`
    * **Input (Tabular):** `Mean_Valence_H1: 0.50, Mean_Arousal_H1: 0.20`
    * **Predicted Output:** `Change_Valence: +0.35, Change_Arousal: +0.65`

## 2. Code Structure

The submission is organized into three separate folders, each containing the end-to-end solution (Training $\rightarrow$ Inference) for the respective subtask.
```text
SEMEVAL2026FT/
├── Subtask1/   # End-to-end training & inference
├── Subtask2a/  # End-to-end training & inference
├── Subtask2b/  # End-to-end training & inference
└── requirements.txt
```

## How to Run (Instructions)

Since the entire pipeline is contained within Jupyter Notebooks, reproducing the results is straightforward.

**Step 1: Environment Setup**

Ensure your local machine has Python 3.10+ installed. We recommend creating a virtual environment to prevent dependency conflicts.

Open your terminal at the project root directory (SEMEVAL2026FT/).

Install all required libraries using the provided requirements file:

```Bash
pip install -r requirements.txt
```
## Results

The performance of the system was evaluated across three subtasks using **Pearson Correlation ($r$)** and **Mean Absolute Error (MAE)**. 

The MoE (Mixture of Experts) architecture was implemented with a sliding window logic (Window Size = 4 for Subtask 1) to process the data.

### Performance Metrics & Official Rankings

The table below summarises our official performance and rankings in SemEval-2026 Task 2.

| Subtask | Outcome | Primary Metric ($r$) | MAE | **Official Rank** |
| :--- | :--- | :--- | :--- | :--- |
| **Subtask 1** | Valence (V) | 0.636688 | 0.693666 | **9th** / 26 |
| | Arousal (A) | 0.488985 | 0.407814 | |
| **Subtask 2a** | Valence (V) | 0.629211 | 1.141477 | **5th** / 15 |
| | Arousal (A) | 0.632732 | 0.688843 | |
| **Subtask 2b** | Valence (V) | -0.168636 | 0.722768 | 11th / 12 |
| | Arousal (A) | -0.059553 | 0.568374 | |

### 5.2. Key Findings

* **Subtask 1 (Static/Batch):** The model achieved a strong composite correlation for Valence ($r \approx 0.64$), while Arousal showed moderate correlation ($r \approx 0.49$).
* **Subtask 2a:** This task yielded the most balanced results, with high correlation scores exceeding 0.62 for both dimensions.
* **Subtask 2b:** The model encountered challenges with this specific subtask, resulting in negative correlation values. This suggests a discrepancy between the predicted trends and the ground truth for this subset.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e695bfab-d14b-4f7a-9f04-573a9e95c1fc" width="100%" alt="Subtask 2A Training Metrics">
  <br>
  <em>Figure 1: Training/Validation Loss (left) and Pearson Correlation (Centre) for The Best Subtask 2A: State Change Forecasting.</em>
</p>

## 📖 Citation
If you use this system or our approach in your research, please cite our paper:

```bibtex
@inproceedings{phuong-etal-2026-citd,
    title = "{CITD}@{UIT} at {S}em{E}val-2026 Task 2: Temporal Mixture-of-Experts for Longitudinal Valence and Arousal Prediction from Ecological Essays",
    author = "Phuong, Son The  and Ngo, My Thuy-Tra  and Minh Dao, Tri  and Nguyen, Duc-Vu",
    booktitle = "Proceedings of the 20th International Workshop on Semantic Evaluation (2026)",
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2026.semeval-1.25/](https://aclanthology.org/2026.semeval-1.25/)",
    doi = "10.18653/v1/2026.semeval-1.25"
}
```

## Author & Contact

Team Leader & Primary Developer:

- Son The Phuong

- University of Information Technology (UIT), VNU-HCM.

Contact for Correspondence:

- Email: phuongtheson22020210@gmail.com

- Role: Team Leader, responsible for model architecture design, training, and system submission.





