import pandas as pd
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import random
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import torch

import src.dataloader as dataloader
import src.models as models
import src.utils as utils

# TODO: REMOVE LATER
import smtplib
from email.mime.text import MIMEText

def send_email_notification():
    msg = MIMEText("Your Python script has completed execution!")
    msg['Subject'] = "Script Finished"
    msg['From'] = "troschkop@gmail.com"
    msg['To'] = "arianiseni76@gmail.com"

    # Gmail SMTP configuration
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Login credentials
    your_email = "troschkop@gmail.com"
    your_password = "qozw vnbp ppvm bcbz"  # Use your Gmail App Password

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade connection to secure
            server.login(your_email, your_password)
            server.send_message(msg)
        print("Notification email sent successfully!")
    except Exception as e:
        print(f"Failed to send notification email: {e}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load the dataset
    print("Loading dataset...")
    dataset_raw = dataloader.dataset
    df_train_raw = pd.DataFrame(dataset_raw['train']).head(1000)
    df_test_raw = pd.DataFrame(dataset_raw['test']).head(1000)

    sizes = [50, 100, 200, 400, 600, 800, 1000]

    # Create specific splits
    nested_splits = utils.create_specific_splits(df_train_raw, sizes)
    nested_splits = {key: Dataset.from_pandas(value)
                     for key, value in tqdm(nested_splits.items(), desc="Converting splits to Dataset")}

    # Tokenize the data
    nested_splits_tokenized = {key: value.map(utils.tokenize_function, batched=True)
                               for key, value in tqdm(nested_splits.items(), desc="Tokenizing data")}

    test_tokenized = (dataset_raw['test'].select(range(1000))
                      .map(utils.tokenize_function, batched=True)
                      .with_format("torch"))

    training_args = TrainingArguments(
        output_dir="outputs/results",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=3.227478802173934e-05,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="outputs/logs",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    losses_per_epoch = {}
    results = {}

    for size, train_dataset in tqdm(nested_splits_tokenized.items(), desc="Training models"):

        trainer = Trainer(
            model=models.model,
            args=training_args,
            train_dataset=train_dataset.with_format("torch"),
            eval_dataset=test_tokenized,
            processing_class=models.tokenizer,
            compute_metrics=utils.compute_metrics
        )

        print(f"Training model on subset: {size}.")
        print("-"*50)
        trainer.train()
        evaluation_results = trainer.evaluate()
        results[size] = evaluation_results
        print(f"Evaluation results: {evaluation_results}")
        print("-"*50)

        train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
        losses_per_epoch[size] = train_loss

        model_output_dir = f"outputs/results/{size}_model"
        tokenizer_output_dir = f"outputs/results/{size}_tokenizer"
        models.model.save_pretrained(model_output_dir)
        models.tokenizer.save_pretrained(tokenizer_output_dir)
        print(f"Model and tokenizer saved in {model_output_dir} and {tokenizer_output_dir} respectively.")
        print("Training completed.")

    # Print all results
    print(f"All Evaluation results: {results}")

    # Plot the losses
    fig, ax = plt.subplots()
    for size, losses in tqdm(losses_per_epoch.items(), desc="Plotting losses"):
        ax.plot(losses, label=size)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.title("Training Loss")
    plt.show()

    send_email_notification()


if __name__ == "__main__":
    main()

