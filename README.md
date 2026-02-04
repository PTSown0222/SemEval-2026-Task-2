# SemEval - 2026 Task 2: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays - System Submission

**Team Name:** Student Of University Information Of Technology (UIT)

## 1. System Description

### A. Data Strategy

**Source:** We strictly utilized the official SemEval 2026 Task 2 dataset (Training, Validation, and Testing splits).

Data Integrity: **Data Integrity:** We enforced strict separation between training and validation sets to ensure no data leakage. The chronological order of user posts was preserved by sorting `timestamp` and `user_id` before processing.

**Subtask-Specific Preprocessing:**

* **Subtask 1:** We applied text normalization (spacing correction) and a sliding window mechanism ($k=4$) to incorporate user history context. The data splitting logic is implemented in the initialization cells of the notebook.

* **Subtask 2A:** We applied text normalization (spacing correction) and a sliding window mechanism ($k=8$) to incorporate user history context.

* **Subtask 2B:** We model the user's emotional shift by comparing their past (Group 1) and recent (Group 2) activities using a **Dual-Sequence Strategy**:
    * *Text:* Concatenation of the last 5 posts from Group 1 and Group 2 with a separator structure: `<s>` G1_Posts `</s>` `</s>` G2_Posts `</s>`.
    * *Tabular:* Incorporation of historical baselines (`mean_valence_half1`, `mean_arousal_half1`) as auxiliary numerical features.

### B. Model Architecture

**Backbone:** We utilized **`cardiffnlp/twitter-roberta-base-sentiment-latest`** as the core encoder for all subtasks.

##### Subtask 1: Longitudinal Affect Assessment

- Architecture: RoBERTa Cardiff + Mean Pooling + Soft Gating Mixture-of-Experts (MoE) Head (4 Experts).

- Inputs: A sequence of text combining past posts ($k=4$) and the current post, separated by `</s>`. Max length: 384 tokens.

- Outputs: Two continuous scores: Valence and Arousal.

#### Subtask 2A: Forecasting Future Variation in Affect - State change

- Architecture: RoBERTa Cardiff + Attention Pooling + Sparse TopK Mixture-of-Experts (MoE) Head (4 Experts).

- Outputs: Two continuous scores representing State Change Valence and State Change Arousal.

#### Subtask 2B: Forecasting Future Variation in Affect - Dispositional change

- Architecture: RoBERTa Cardiff + Mean Pooling + Sparse TopK Mixture-of-Experts (MoE) Head (4 Experts).

- Mechanism: The model fuses text embeddings (768-dim) with numerical baseline features (2-dim) before passing through a Sparse MoE Head (Top-2 Gating) to predict the magnitude of emotional change.

- Outputs: Two continuous scores representing Disposition Change Valence and Disposition Change Arousal.

### C. Training Strategy

We optimized the models using the Concordance Correlation Coefficient (CCC) Loss to maximize the correlation between predictions and ground truth.

* **Subtask 1 Loss:** `0.4 * Loss_Valence + 0.6 * Loss_Arousal`
    * Learning Rate: `2e-5` | Max Seq Length: `384`

* **Subtask 2A Loss:** `0.5 * Loss_Valence + 0.5 * Loss_Arousal`
    * Learning Rate: `2e-5` | Max Seq Length: `512`

* **Subtask 2B Loss:** `0.5 * Loss_Valence + 0.5 * Loss_Arousal`
    * Learning Rate: `3e-5` | Max Seq Length: `512`

### D.Examples

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
│
├── Subtask1/
│   ├── data/
│   ├── weights/                            <-- Download from section 3 (current is empty)
│   ├── Inference_subtask1.ipynb            <-- Inference (locally run for Organizers test and export CSV)
│   └── subtask1-code.ipynb                 <-- Including end to end Training + Inference script in Kaggle
│
├── Subtask2a/
│   ├── data/
│   ├── weights/
│   │   └── final_model_subtask2a/          <-- Download from section 3 (current is empty)
│   ├── Inference_sutask2a.ipynb            <-- Inference (locally run for Organizers test and export CSV)
│   └── subtask2a-code.ipynb                <-- Including end to end Training + Inference script in Kaggle
│
├── Subtask2b/
│   ├── data/
│   ├── weights/                            <-- Download from section 3 (current is empty)
│   ├── Inference_sutask2a.ipynb            <-- Inference (locally run for Organizers test and export CSV)                             
│   └── subtask2b-code.ipynb                <-- Including end to end Training + Inference script in Kaggle
│
├── README.md
└── requirements.txt
```

## 3. Download Model Weights

Due to file size limits, the trained model weights are hosted externally. Please download the weights for each subtask and place them in the corresponding folders before running inference.

| Subtask | Download Link | File Name |
| :--- | :--- | :--- |
| **Subtask 1** | **[Download Here](https://drive.google.com/file/d/1MbGPYny-1ukdSQltWZeq6TXKHvWirZfg/view?usp=drive_link)** | `final_model_subtask1.zip` |
| **Subtask 2A** | **[Download Here](https://drive.google.com/file/d/14k8RlNk7rbZ6EjjSfSyGAflLV7PA_PWc/view?usp=drive_link)** | `final_model_subtask2a.zip` |
| **Subtask 2B** | **[Download Here](https://drive.google.com/file/d/1paiXw5siAuEEH1exdkfqS53YPr4vioOn/view?usp=drive_link)** | `final_model_subtask2b.zip` |

## 4. How to Run (Instructions)

Since the entire pipeline is contained within Jupyter Notebooks, reproducing the results is straightforward.

**Step 1: Environment Setup**

Ensure your local machine has Python 3.10+ installed. We recommend creating a virtual environment to prevent dependency conflicts.

Open your terminal at the project root directory (SEMEVAL2026FT/).

Install all required libraries using the provided requirements file:

```Bash
pip install -r requirements.txt
```

**Step 2: Download and Place Model Weights**

Because of the large file sizes, pre-trained weights are hosted on Google Drive.

1. Download the weights for each subtask using the links in Section 3.

2. Extract the contents of each .zip file.

3. Move the extracted folders into the respective weights/ directory of each subtask:

- Subtask 1: Place the folder in Subtask1/weights/final_model_subtask1/.

- Subtask 2A: Place the folder in Subtask2a/weights/final_model_subtask2a/.

- Subtask 2B: Place the folder in Subtask2b/weights/final_model_subtask2b/.

**Step 3: Prepare Test Data**

Place the official test CSV files provided by the organizers into the data/ folder of the appropriate subtask.

Subtask 1: Place test files in Subtask1/data/data_sub1/. The notebook will automatically search all subdirectories to find the correct CSV.

Subtask 2A: Place test files in Subtask2a/data/.

Subtask 2B: Place test files in Subtask2b/data/. Important: For Subtask 2B, the training CSV must also be in this folder so the system can calculate and fit the StandardScaler for numerical features.

**Step 4: Run the Inference Notebooks**

Navigate to the specific subtask folder (e.g., /Subtask1).

Open the notebook named Inference_subtask*.ipynb.

Set the Kernel to your Python environment.

Click "Run All". The system will perform the following actions automatically:

Auto-Detection: Locates the test dataset and the weight files without requiring manual path editing.

Data Processing: Applies the sliding window logic (Window size = 4 for Subtask 1).

Model Loading: Initializes the MoE architecture and loads the weights.

Generation: Runs the model on the device (prioritizes CUDA GPU if detected, otherwise defaults to CPU).

Export: Saves the final results to submission.csv in the subtask folder.

## 5. Author & Contact

Team Leader & Primary Developer:

- Son The Phuong

- University of Information Technology (UIT), VNU-HCM.

Contact for Correspondence:

- Email: 25210032@ms.uit.edu.vn

- Role: Team Leader, responsible for model architecture design, training, and system submission.





