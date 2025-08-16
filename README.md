AI Resume Ranker 📝🤖

A machine learning web app that ranks resumes against a job description using NLP.

🔹 Overview

Recruiters spend hours manually screening resumes. This project automates the process by ranking resumes according to how well they match a given job description (JD).

It uses TF-IDF embeddings + cosine similarity to compute scores, with support for PDF/DOCX parsing, skills extraction, and evaluation using ground-truth labels (accuracy, F1, confusion matrix, ROC/PR curves).

🔹 Features

📄 Parse resumes from:

CSV dataset (Kaggle Resume Dataset).

Local PDF/DOCX files.

🧹 Preprocessing:

Text extraction (pdfminer.six, docx2txt).

Cleaning (lowercasing, stopwords, lemmatization).

🔍 Feature Extraction:

TF-IDF embeddings (baseline).

[Optionally extendable to BERT/Sentence Transformers].

📊 Matching & Ranking:

Cosine similarity with JD embedding.

Resume scores from 0–1.

🏆 Output:

Ranked CSV (rankings.csv).

Artifacts (scores_full.csv, vectorizer.pkl, embeddings.h5, config.yaml).

✅ Evaluation:

Human labeling via labels.csv.

Accuracy vs Threshold graph.

Confusion Matrix heatmap.

ROC & PR curves.

🔹 Dataset

Kaggle Resume Dataset
Path:

C:\Users\sagni\Downloads\Resume Ranker\archive (1)\Resume\Resume.csv


Includes text-formatted resumes with categories.

LiveCareer Resume Examples (PDF/DOCX)
Path:

C:\Users\sagni\Downloads\Resume Ranker\archive (1)\data\data


Useful for testing PDF/DOCX parsing.

🔹 Project Structure
Resume Ranker/
│── archive (1)/
│   ├── Resume/Resume.csv        # Kaggle CSV resumes
│   ├── data/data/               # Resume PDFs/DOCXs
│
│── resume_ranker.py             # Main pipeline script
│── scores_full.csv              # Cached scores for all resumes
│── rankings.csv                 # Top-K ranked resumes
│── labels.csv                   # Ground truth labels (you edit)
│── labels_aligned_preview.csv   # Auto-aligned labels vs scores
│── unmatched_labels.csv         # Unmatched names for debugging
│── config.yaml                  # Run config
│── vectorizer.pkl               # Saved TF-IDF vectorizer
│── embeddings.h5                # Saved embeddings
│── metrics.json                 # Metrics summary
│── README.md                    # (this file)

🔹 Usage
1. Clone repo & install requirements
pip install -r requirements.txt


(or manually: pandas numpy scikit-learn pdfminer.six docx2txt tqdm matplotlib)

2. Run Ranking (Command Line)
python resume_ranker.py --jd "Looking for a Data Scientist with Python, NLP, TensorFlow, AWS, and MLOps." --top_k 20


This will produce:

rankings.csv (top matches).

Saved artifacts (pkl, h5, yaml, json).

3. Run in Jupyter Notebook

Since argparse conflicts with Jupyter args, use the Jupyter wrapper cell version. It sets JD and Top-K defaults automatically.

Example cell:

# Run full pipeline inside Jupyter
%run resume_ranker_notebook_version.py

4. Labeling for Evaluation

A labels.csv template is generated automatically:

id,name,source,score,label
csv_0,John Doe,csv,0.812,
file_3,resume123,pdf,0.547,


Fill the label column with:

1 → relevant/selected

0 → irrelevant/rejected

Save and re-run evaluation.

5. Evaluation (Accuracy & Heatmap)
%run evaluate_ranker.py


Outputs:

Accuracy vs Threshold graph.

Confusion Matrix heatmap.

ROC curve.

Precision-Recall curve.

Preview files:

labels_aligned_preview.csv → check matched IDs/names.

unmatched_labels.csv → names not found in scores.

🔹 Example Output
Resume Rankings
id	name	source	score
csv_23	Alice Smith	csv	0.912
file_3	resume_ds	pdf	0.851
csv_99	John Doe	csv	0.743
Accuracy vs Threshold

Confusion Matrix

ROC Curve

🔹 Tech Stack

Python (pandas, numpy, scikit-learn)

NLP (TF-IDF baseline, extendable to BERT)

Text Extraction: pdfminer.six, docx2txt

Visualization: matplotlib, seaborn

Web UI (future): Flask / Streamlit

🔹 Future Improvements

✅ Use Sentence-BERT embeddings for semantic similarity.

✅ Add skills NER (spaCy or regex-based).

✅ Deploy as a Streamlit app for HR teams.

✅ Multi-label classification (role-specific ranking).

🔹 Credits

Resume Dataset: Kaggle

Resume examples: LiveCareer

Libraries: scikit-learn, pdfminer.six, docx2txt, tqdm, matplotlib
