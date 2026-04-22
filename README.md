# Yelp Review Sentiment Analysis — NLP Project

An end-to-end NLP pipeline for sentiment classification of Yelp business reviews, built using Python, scikit-learn, and NLTK.

## Overview

This project investigates whether the sentiment of a Yelp review (positive or negative) can be predicted from its text content. Using a dataset of ~200,000 Yelp reviews, we build and compare multiple classification models, perform keyword analysis, and visualize review language patterns through word clouds.

**Task:** Binary sentiment classification
- Positive: 4–5 star reviews
- Negative: 1–2 star reviews
- Neutral (3 star) reviews were excluded due to ambiguity

**Best result:** SVM with TF-IDF features — 95.8% weighted F1 score

---

## Project Structure

```
NLP/
├── data/
│   ├── ratings.csv                  # Raw merged review + business dataset
│   └── ratings_clean.parquet        # Preprocessed, cleaned dataset
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   └── 02_modeling.ipynb            # Model training, evaluation, and analysis
├── src/
│   └── preprocessing.py             # Preprocessing pipeline script
├── requirements.txt
└── README.md
```

---

## Dataset

The raw data was sourced from the [Yelp Open Dataset](https://www.yelp.com/dataset), combining the reviews and business JSON files. The merged dataset contains:

- `review_id` — unique review identifier
- `review_rating` — star rating (1–5)
- `review` — raw review text
- `review_length` — word count
- `categories` — business category tags

---

## Preprocessing Pipeline

Implemented in `src/preprocessing.py`:

1. Removed duplicate reviews and null values
2. Filtered reviews to 10–500 words
3. Detected and kept English-only reviews using `langdetect`
4. Lowercased and removed URLs, HTML tags, punctuation
5. Removed stopwords (preserving negation words like "not", "never", "no")
6. Lemmatized tokens using NLTK `WordNetLemmatizer`
7. Saved cleaned dataset as Parquet for efficient loading

---

## Models

All models used TF-IDF vectorization (`max_features=10,000`, `ngram_range=(1,2)`) with `class_weight='balanced'` to handle the 75/25 class imbalance.

| Model | Negative F1 | Positive F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 0.913 | 0.970 | 0.955 |
| Naive Bayes | 0.852 | 0.954 | 0.929 |
| **SVM (LinearSVC)** | **0.918** | **0.972** | **0.958** |

**SVM** achieved the best performance across all metrics. Naive Bayes underperformed on the negative class due to its word independence assumption, which fails to capture negation phrases like "not good".

---

## Key Findings

- Top positive predictors: `great`, `delicious`, `amazing`, `perfect`, `awesome`
- Top negative predictors: `worst`, `horrible`, `bland`, `disappointing`, `rude`
- Negation bigrams (`not good`, `not worth`, `definitely not`) were correctly identified as strong negative signals, validating the decision to preserve negation words during preprocessing
- Positive reviews tended to use more evaluative adjectives, while negative reviews used more descriptive nouns — consistent with research showing negative reviewers recount facts while positive reviewers express emotions

---

## Setup

```bash
# Clone the repo
git clone https://github.com/spennyfinn/Yelp_Review_NLP.git
cd Yelp_Review_NLP

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The raw data files are not included in this repo due to size. Download the Yelp Open Dataset from [yelp.com/dataset](https://www.yelp.com/dataset) and place the JSON files in the `data/` directory.

---

## Future Work

- **BERTopic** for more robust topic modeling (LDA produced uninterpretable results due to preprocessed bigrams in the text)
- **DistilBERT** for transformer-based sentiment classification to push accuracy beyond the TF-IDF ceiling
- **Aspect-based sentiment analysis** to separately score food, service, and atmosphere within each review

---

## Tech Stack

- Python 3.13
- pandas, numpy
- scikit-learn
- NLTK
- matplotlib, seaborn
- wordcloud
- langdetect
- pyarrow (Parquet I/O)
