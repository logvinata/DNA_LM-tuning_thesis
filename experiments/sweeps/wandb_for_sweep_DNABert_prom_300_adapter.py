#!/usr/bin/env python
# coding: utf-8

# Install and import

import os
import random
import sys

import evaluate as eval
import numpy as np
import torch
import wandb
from datasets import load_dataset
# from sklearn import metrics
from transformers import (
    AdapterConfig,
    AutoConfig,
    AutoTokenizer,
    BertModelWithHeads,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("NO CUDA!!!")
    sys.exit(1)


#  Drive, paths and config
data_path = "../datasets/promoters/promoters_300/"
split_path = data_path + "train-val-test_split/"

# additional parameters for adapters phm ( parameterized hypercomplex multiplication layers)
# used by default in this version of HF adapters for pfiefer adapters
# see https://arxiv.org/pdf/2106.04647.pdf
phm_param = {
    "phm_layer": False,
    "phm_dim": 4,
    "factorized_phm_W": True,
    "shared_W_phm": False,
    "shared_phm_rule": True,
    "factorized_phm_rule": False,
    "phm_c_init": "normal",
    "phm_init_range": 0.0001,
    "learn_phm": True,
    "hypercomplex_nonlinearity": "glorot-uniform",
    "phm_rank": 1,
    "phm_bias": True,
}


# I only use one type of adapters in this sweep for simplicity and clarity of parameters configuration
parameters = {
    "seed": 0,
    # 'model_name': 'zhihan1996/DNA_bert_6',
    "project_name": "DNA_BERT_sweeps_prom_300",
    "task_name": "prom_300",
    "num_train_epochs": 10,
    "save_total_limit": 7,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "metric_for_best_model": "matthews_correlation",
    "early_stopping_patience": 5,
    # add adapter_parameters
    "reduction_factor": 16,
    "non_linearity": "swish",
    "original_ln_before": True,
    "original_ln_after": True,
    "phm_param": phm_param,
}

# the model is really unstable
# this has to be set before loading the model and tokenizer
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


#  Load and Construct model and tokenizer
def load_model():
    model_config = AutoConfig.from_pretrained(
        "zhihan1996/DNA_bert_6", num_labels=2, trust_remote_code=True
    )

    model = BertModelWithHeads.from_pretrained(
        "zhihan1996/DNA_bert_6", config=model_config, trust_remote_code=True
    )

    return model_config, model


# Add Adapter
def add_adapter(model, parameters):
    # Define an adapter configuration
    adapter_config = AdapterConfig.load(
        "pfeiffer",
        reduction_factor=parameters["reduction_factor"],
        non_linearity=parameters["non_linearity"],
        original_ln_before=parameters["original_ln_before"],
        original_ln_after=parameters["original_ln_after"],
    )
    print(f"adapter_config : {adapter_config}")
    # Add a new adapter
    model.add_adapter("prom_300", config=adapter_config)
    # Add a matching classification head
    # should have the same name as adapter
    model.add_classification_head("prom_300", num_labels=2, overwrite_ok=True)
    # Activate the adapter and the head
    model.train_adapter("prom_300")
    # print(model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"number of trainable parameters: {num_trainable_params / 1000000}, \ntotal number of parameters: {model.num_parameters() / 1000000}"
    )

    return model


# Dataset

# Start with loading kmers


def load_data(small=False):
    train_6mers = load_dataset(
        "csv",
        data_files=split_path + "6_mers/hg38_len_300_prom_train_6mers.csv",
        delimiter=",",
    )["train"]
    val_6mers = load_dataset(
        "csv",
        data_files=split_path + "6_mers/hg38_len_300_prom_val_6mers.csv",
        delimiter=",",
    )["train"]

    test_6mers = load_dataset(
        "csv",
        data_files=split_path + "6_mers/hg38_len_300_prom_test_6mers.csv",
        delimiter=",",
    )["train"]
    if small:
        small_train_6mers = train_6mers.shuffle(seed=42).select(range(10))
        small_val_6mers = val_6mers.shuffle(seed=42).select(range(3))
        small_test_6mers = test_6mers.shuffle(seed=42).select(range(3))
        return small_train_6mers, small_val_6mers, small_test_6mers

    return train_6mers, val_6mers, test_6mers


# Tokenize
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


def tokenize_data(train_6mers, val_6mers, test_6mers, tokenization=tokenization):
    train = train_6mers.map(tokenization, batched=True, batch_size=len(train_6mers))
    val = val_6mers.map(tokenization, batched=True, batch_size=len(val_6mers))
    test = test_6mers.map(tokenization, batched=True, batch_size=len(test_6mers))
    print(train, val, test)
    return train, val, test


# define metrics
# accuracy = eval.load("accuracy")
# precision = eval.load("precision")
# recall = eval.load("recall")
# f1 = eval.load("f1")
# mcc = eval.load("matthews_correlation")
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
    metrics_scores = combined_metrics.compute(predictions=predictions, references=labels)

    return metrics_scores


def save_all(run, trainer, test):
    # trainer.save_model("./results/" + model_name)

    # make predictions on test and compute metrics
    predictions = trainer.predict(test)
    print(predictions[2])

    #     # save metrics and predictions on disk
    #     # predictions
    #     df = pd.DataFrame(data=predictions[0])
    #     df["label"]=predictions[1]
    #     probs = torch.nn.functional.softmax(torch.from_numpy(predictions[0]), dim=-1)
    #     df["probability_0"] = probs[:, 0]
    #     df["probability_1"] = probs[:, 1]
    #     df["prediction"] = np.where(df["probability_0"] >= 0.5, 0, 1)
    #     #wand table requires column names not to be integers
    #     df.rename(columns = {1 : '1', 0 : '0'}, inplace = True)
    #     df.to_csv("./results/" + model_name + ".csv")

    #     # metrics
    #     metrics_file = "./results/" + model_name + "_metric.txt"
    #     with open(metrics_file, 'w') as f:
    #         for key, value in predictions[2].items():
    #             f.write(f'{key} : {value}\n')

    # log metrics and predictions on wandb
    # note: don't log preds for sweep
    # !!! REVIEW FOR SWEEP

    #     # predictions
    #     description="promotors 300 predictions with probabilities from model after run " + model_name
    #     prob_table = wandb.Table(dataframe=df)
    #     model_pred = wandb.Artifact(
    #         "DNABERT_adapter_prom_300_pred",
    #         type="adapters predictions on test",
    #         description=description
    #     )
    #     model_pred.add(wandb.Table(dataframe=df), description)
    #     run.log_artifact(model_pred, aliases=['adapter tuning predictions', 'baseline'])

    #     # metrics
    #     columns = ['metric', 'test value']
    #     data = []
    #     for key, value in predictions[2].items():
    #         data.append([key, value])
    #     description="promotors 300 metrics from model after run " + model_name
    #     model_metrics = wandb.Artifact(
    #         "DNABERT_adapter_prom_300_metrics",
    #         type="adapters metrics on test",
    #         description=description
    #     )
    #     model_metrics.add(wandb.Table(columns=columns, data=data), description)
    #     run.log_artifact(model_metrics, aliases=['adapter tuning metrics'])
    wandb.log(predictions[2])

    # add metrics to summary
    for key, val in predictions[2].items():
        key_name = key + ".best"
        run.summary[key_name] = val
        print(key_name, "added to summary")

    # confusion matrix


#     confusion_matrix = metrics.confusion_matrix(df['label'], df['prediction'], labels=[0, 1])
#     wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
#                         preds=df['prediction'], y_true=df['label'],
#                         class_names=[False, True])})


#     # and finally model seems to be not logged automatically
#     model=wandb.Artifact(model_name, type="model",
#                      description="baseline")
#     run.log_artifact(model, aliases=['adapter tuning', 'baseline', model_name])


def training_script(parameters=parameters, small=True, log=True):
    # set env variables for wandb
    # WANDB_WATCH all doesn't really work anyway
    os.environ["WANDB_LOG_MODEL"] = "end"
    # os.environ['WANDB_WATCH'] = 'all'
    set_seed(parameters["seed"])

    # init wandb run
    if small:
        note = "test run on a micro dataset"
        # model_name = "logging_test"
    else:
        note = "prom_300 sweep"
    run = wandb.init(
        project="DNA_BERT_sweeps_prom_300",
        job_type="adapter_train_hyperoptimization",
        config=parameters,
        save_code=True,
        notes=note,
    )
    parameters = wandb.config
    print("parameters: ", parameters)
    # log the data usage
    run.use_artifact(
        "logvinata/DNA_BERT_prom_300/prom_300_splitted_6_mers:v0", type="6mers"
    )
    run.log_code(
        ".",
        include_fn=lambda path: "./DNA_BERT / wandb_for_sweep_DNABert_prom_300_adapter.py",
    )

    # load model and tokenizer
    model_config, model = load_model()
    print("model_config", model_config)
    # add adapter and head
    model = add_adapter(model, parameters)

    # load and tokenize data
    train_6mers, val_6mers, test_6mers = load_data(small=small)
    train, val, test = tokenize_data(train_6mers, val_6mers, test_6mers)

    # define metrics
    accuracy = eval.load("accuracy")
    precision = eval.load("precision")
    recall = eval.load("recall")
    f1 = eval.load("f1")
    mcc = eval.load("matthews_correlation")

    # set TrainingArguments for Trainer
    if small:
        num_epochs = 2
    else:
        num_epochs = parameters["num_train_epochs"]

    training_args = TrainingArguments(
        report_to="wandb",
        output_dir="./results",
        num_train_epochs=num_epochs,
        # save_steps=5921,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=parameters["save_total_limit"],
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        weight_decay=parameters["weight_decay"],
        learning_rate=parameters["learning_rate"],
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        warmup_ratio=parameters["warmup_ratio"],
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        label_names=[
            "labels"
        ],  # see https://github.com/huggingface/transformers/issues/22885
        metric_for_best_model=parameters["metric_for_best_model"],
    )

    # add earlystopping
    early_stop = [
        EarlyStoppingCallback(
            early_stopping_patience=parameters["early_stopping_patience"]
        )
    ]

    # finally trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        callbacks=early_stop,
    )

    # train
    trainer.train()

    if log:
        save_all(run, trainer, test)

    # return model

    run.finish()


training_script(small=False, log=True)
