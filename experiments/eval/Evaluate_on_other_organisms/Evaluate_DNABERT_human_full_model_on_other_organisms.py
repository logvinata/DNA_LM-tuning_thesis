# Import

import csv
import os
import random
import sys

import datasets
import evaluate as eval
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
import wandb
from datasets import load_dataset
from sklearn import metrics
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("NO CUDA!!!")
    sys.exit(1)


# Set up and parameters


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_data(small=False):
    #     train_6mers = load_dataset('csv',
    #                          data_files=split_path + f"6_mers/hg38_{task_name}_train_6mers.csv",
    #                          delimiter=",")['train']
    #     val_6mers = load_dataset('csv',
    #                          data_files=split_path + f"6_mers/hg38_{task_name}_val_6mers.csv",
    #                          delimiter=",")['train']

    test_6mers = load_dataset(
        "csv", data_files=k_mers_path + f"{dataset_name}_test_6mers.csv", delimiter=","
    )["train"]
    if small:
        #         small_train_6mers = train_6mers.shuffle(seed=42).select(range(2000))
        #         small_val_6mers = val_6mers.shuffle(seed=42).select(range(200))
        small_test_6mers = test_6mers.shuffle(seed=42).select(range(200))
        return (small_test_6mers,)  # small_train_6mers, small_val_6mers,

    return test_6mers  # train_6mers, val_6mers,


def tokenization(sequences_batch):
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNA_bert_6", do_lower_case=False, trust_remote_code=True
    )
    return tokenizer(
        sequences_batch["kmer"],
        max_length=512,  # max len of BERT
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True
    )
    return model.config, model


def compute_metrics(val_pred):
    logits, labels = val_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = eval.load("accuracy")
    precision = eval.load("precision")
    recall = eval.load("recall")
    f1 = eval.load("f1")
    mcc = eval.load("matthews_correlation")

    # Combine metrics
    combined_metrics = eval.combine([accuracy, precision, recall, f1, mcc])

    # Compute metrics
    metrics_scores = combined_metrics.compute(
        predictions=predictions, references=labels
    )

    return metrics_scores


if __name__ == "__main__":
    set_seed(0)
    x = [
        f.name
        for f in os.scandir(f"../../datasets/promoters/EPD_data/EPD_data_csv")
        if f.is_file()
    ]
    for f in x:
        organism, genome, task, length, _ = f.split("_")
        print(organism, task, genome, length)
        dataset_name = f"{organism}_{genome}_{task}_{length}"
        notebook_name = (
            f"DNABERT_human_full_evaluate_on_{dataset_name}"  # just convenience
        )
        project_name = f"evaluate_{task}_{length}_datasets"
        model_name = "DNABERT_human"
        group = model_name
        tag = organism

        # paths to data
        data_path = f"../../datasets/promoters/{organism}_{task}_{length}"
        split_path = data_path + "/train-val-test_split/"
        k_mers_path = split_path + "6_mers/"
        # path to model if local
        model_path = "../results/DNABERT_prom_300_f_2"

        dataset_art_name = (
            f"logvinata/{task}_{length}_datasets/{dataset_name}_splitted_6_mers:latest"
        )
        model_art_name = "logvinata/model-registry/DNA_BERT_full_prom_300:v2"

        # OLD VERSION
        parameters = {
            "seed": 0,
            "model_name": "zhihan1996/DNA_bert_6",
            "task_name": "prom_300",
            "num_train_epochs": 10,
            "save_total_limit": 7,
            "weight_decay": 0.02,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "metric_for_best_model": "matthews_correlation",
            "early_stopping_patience": 5,
        }
        # OLD VERSION
        training_args = TrainingArguments(
            report_to="wandb",
            output_dir="./results/" + notebook_name,
            num_train_epochs=parameters["num_train_epochs"],
            # save_steps=5921,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=parameters["save_total_limit"],
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            weight_decay=parameters["weight_decay"],
            learning_rate=parameters["learning_rate"],
            per_device_eval_batch_size=256,  # !pay attention, this is just for inference
            per_device_train_batch_size=256,
            warmup_ratio=parameters["warmup_ratio"],
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
            label_names=[
                "labels"
            ],  # see https://github.com/huggingface/transformers/issues/22885
            metric_for_best_model=parameters["metric_for_best_model"],
        )

        # wandb

        run = wandb.init(
            project=project_name,
            job_type="evaluation",
            name=f"{model_name}_on_{genome}",
            config=parameters,
            save_code=True,
            notes=notebook_name,
            group=organism,
            tags=[model_name, organism],
        )

        # lineage tracking
        dataset_art = run.use_artifact(dataset_art_name, type="6mers")
        model_art = run.use_artifact(model_art_name, type="model")

        # dataset

        test_6mers = load_data(small=False)
        test = test_6mers.map(tokenization, batched=True, batch_size=10000)
        model_config, model = load_model()
        print(model_config, model)  # this will be in the logs
        print("run config:", run.config)
        print("training_args:", training_args)

        # evaluate and log

        # define metrics
        accuracy = eval.load("accuracy")
        precision = eval.load("precision")
        recall = eval.load("recall")
        f1 = eval.load("f1")
        mcc = eval.load("matthews_correlation")

        # finally trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
        )

        predictions = trainer.predict(test)

        # metrics
        print(predictions[2])
        wandb.log(predictions[2])

        # add metrics to summary
        for key, val in predictions[2].items():
            key_name = key + ".best"
            run.summary[key_name] = val
            print(key_name, "added to summary")

        # add model name and organism - can't add to summary, only numbers
        run.log({"model_name": model_name, "dataset": dataset_name})

        # for confusion matrix
        df = pd.DataFrame(data=predictions[0])
        df["label"] = predictions[1]
        probs = torch.nn.functional.softmax(torch.from_numpy(predictions[0]), dim=-1)
        df["probability_0"] = probs[:, 0]
        df["probability_1"] = probs[:, 1]
        df["prediction"] = np.where(df["probability_0"] >= 0.5, 0, 1)
        # wand table requires column names not to be integers
        df.rename(columns={1: "1", 0: "0"}, inplace=True)
        # df.to_csv("./results/" + task_name + ".csv")

        # confusion matrix
        confusion_matrix = metrics.confusion_matrix(
            df["label"], df["prediction"], labels=[0, 1]
        )
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=df["prediction"],
                    y_true=df["label"],
                    class_names=[False, True],
                )
            }
        )

        # log predictions
        model_pred = wandb.Artifact(
            f"predictions_of_{model_name}_on_{genome}",
            type="evaluation on test",
            description=notebook_name,
        )
        model_pred.add(wandb.Table(dataframe=df), notebook_name)
        run.log_artifact(model_pred)

        run.finish()  # we are in a loop
