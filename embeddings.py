import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import src.dataloader as dataloader
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import src.models as models
import src.utils as utils


def create_tsne_map_with_legend_and_save(embeddings_file, labels, size, prefix):

    embeddings = np.load(embeddings_file)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # split points by label
    positiv_points = reduced_embeddings[np.array(labels) == 1]
    negativ_points = reduced_embeddings[np.array(labels) == 0]

    # plot
    plt.figure(figsize=(10, 10))
    plt.scatter(
        positiv_points[:, 0], positiv_points[:, 1], color="blue", label="Positive"
    )
    plt.scatter(
        negativ_points[:, 0], negativ_points[:, 1], color="red", label="Negative"
    )
    plt.title(f"t-SNE map of embeddings for size {size}")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend(loc="best")

    file_name = f"outputs/tsne/{prefix}_tsne_size_{size}.png"
    plt.savefig(file_name)
    plt.show()
    print(f"Saved t-SNE plot for size {size} in {file_name}")


def embed_and_save(samples, prefix):

    for size, dataset in tqdm(samples.items(), desc="Embedding data"):

        content_list = dataset["content"]

        embeddings = models.embedding_model.encode(content_list, show_progress_bar=True)

        file_name = f"outputs/embeddings/{prefix}_embeddings_size_{size}.npy"
        np.save(file_name, embeddings)


def main():

    embedding_model = models.embedding_model

    # Load the dataset
    print("Loading dataset...")
    dataset_raw = dataloader.dataset
    df_train_raw = pd.DataFrame(dataset_raw["train"]).head(1000)

    sizes = [50, 100, 200, 400, 800, 1000, 6000]  # Added 6000 for the full dataset
    nested_splits = utils.create_specific_splits(df_train_raw, sizes)
    nested_splits = {
        key: Dataset.from_pandas(value)
        for key, value in tqdm(
            nested_splits.items(), desc="Converting splits to Dataset"
        )
    }

    nested_splits_tokenized = {
        key: value.map(utils.tokenize_function, batched=True)
        for key, value in tqdm(nested_splits.items(), desc="Tokenizing data")
    }

    print(f"type of nested_splits_tokenized: {type(nested_splits_tokenized)}")
    print(f"nested_splits_tokenized: {nested_splits_tokenized}")

    embed_and_save(nested_splits_tokenized, "train")

    # load the embeddings and plot t-SNE
    for size, dataset in tqdm(
        nested_splits_tokenized.items(), desc="Creating t-SNE plots"
    ):
        labels = dataset["label"]

        embeddings_file = f"outputs/embeddings/train_embeddings_size_{size}.npy"

        create_tsne_map_with_legend_and_save(embeddings_file, labels, size, "train")


if __name__ == "__main__":
    main()
