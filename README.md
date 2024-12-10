# Sentiment Analysis Project

This repository contains the implementation of a sentiment analysis pipeline for binary classification (positive/negative sentiment) using the [Amazon Polarity dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity). The project explores various techniques, including weak labeling, fine-tuning pre-trained models, and evaluating the impact of training data size on model performance.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Preparation](#data-preparation)
3. [Baseline Model](#baseline-model)
4. [Weak Labeling](#weak-labeling)
5. [Combined Training](#combined-training)
6. [Visualization and Evaluation](#visualization-and-evaluation)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Future Enhancements](#future-enhancements)

---

## Project Overview
The primary goal of this project is to:
- Train a binary sentiment classification model using a pre-trained transformer.
- Experiment with weak labeling techniques to expand the training dataset with minimal manual annotation.
- Evaluate model performance under different training setups, including:
  - Baseline training with manual labels.
  - Combined training with manual and weak labels.
- Visualize results and analyze the impact of weak labeling on model accuracy and efficiency.

---

## Data Preparation
1. **Dataset**: We used the [Amazon Polarity dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity) with labeled reviews categorized as positive or negative.
2. **Preprocessing**:
   - Tokenization using Hugging Face's `AutoTokenizer`.
   - Hierarchically nested splits to simulate varying amounts of labeled data.
   - Truncation of reviews exceeding the model's maximum token length.

---

## Baseline Model
1. **Model**: Fine-tuned the `distilbert-base-uncased` transformer model from Hugging Face.
2. **Training**:
   - Evaluated performance using metrics such as accuracy, F1-score, precision, and recall.
   - Saved the model and tokenizer for future use.

---

## Weak Labeling
1. Generated embeddings for labeled and unlabeled data using `sentence-transformers`.
2. Assigned weak labels to unlabeled data using cosine similarity between embeddings.
3. Evaluated the quality of weak labels by comparing them with ground truth labels.

---

## Combined Training
1. Combined the weak labels with manual labels to create an expanded training dataset.
2. Retrained the sentiment classification model with the combined dataset.
3. Compared the model's performance to the baseline setup to evaluate the impact of weak labeling.

---

## Visualization and Evaluation
1. Plotted learning curves to analyze training loss across epochs.
2. Compared performance metrics between the baseline and combined models.
3. Visualized text embeddings using t-SNE for a qualitative analysis of data distribution.

---

## Installation
### Prerequisites
- Python 3.8+
- GPU-enabled environment (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
