import optuna
from transformers import TrainingArguments, Trainer
import utils as utils
import models as models
import dataloader as dataloader


def objective(trial):

    train_dataset_subset = dataloader.dataset["train"].select(range(6000))
    test_dataset_subset = dataloader.dataset["test"].select(range(1000))

    train_tokenized = train_dataset_subset.map(models.tokenizer.tokenize_function, batched=True).with_format("torch")
    test_tokenized = test_dataset_subset.map(models.tokenizer.tokenize_function, batched=True).with_format("torch")

    train_dataset = train_tokenized.with_format("torch")
    test_dataset = test_tokenized.with_format("torch")

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 2, 5)

    training_args = TrainingArguments(
        output_dir="./results_optuna",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs_optuna",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=models.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=models.tokenizer,
        compute_metrics=utils.compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Run 10 trials (you can increase this)

# Get the best hyperparameters
best_hyperparams = study.best_params
print("Best Hyperparameters:", best_hyperparams)

