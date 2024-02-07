#!/usr/bin/env python
# coding: utf-8

# ## Drive, paths and config

import csv
import os
import random
import sys

import datasets
import evaluate as eval
# Import
import fusion_tasks
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
import wandb
from datasets import load_dataset
from sklearn import metrics
from transformers import (AdapterConfig, AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          BertModelWithHeads, EarlyStoppingCallback, Trainer,
                          TrainingArguments)
from transformers.adapters.composition import Fuse

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("NO CUDA!!!")
    sys.exit(1)

# util functions
# from train_utils import (
#     set_seed,
#     load_data,
#     tokenization,
#     load_adapter_model,
#     add_adapter,
#     compute_metrics,
#     save_all,
# )


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
from transformers import (AdapterConfig, AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          BertModelWithHeads, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("NO CUDA!!!")
    sys.exit(1)


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
    train_6mers = load_dataset(
        "csv", data_files=k_mers_path + f"{dataset_name}_train_6mers.csv", delimiter=","
    )["train"]
    val_6mers = load_dataset(
        "csv", data_files=k_mers_path + f"{dataset_name}_val_6mers.csv", delimiter=","
    )["train"]

    test_6mers = load_dataset(
        "csv", data_files=k_mers_path + f"{dataset_name}_test_6mers.csv", delimiter=","
    )["train"]
    if small:
        #         small_train_6mers = train_6mers.shuffle(seed=42).select(range(2000))
        #         small_val_6mers = val_6mers.shuffle(seed=42).select(range(200))
        small_test_6mers = test_6mers.shuffle(seed=42).select(range(200))
        return (small_test_6mers,)  # small_train_6mers, small_val_6mers,

    return train_6mers, val_6mers, test_6mers


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
    print("started tokenization")
    train = train_6mers.map(tokenization, batched=True, batch_size=10000)
    val = val_6mers.map(tokenization, batched=True, batch_size=10000)
    test = test_6mers.map(tokenization, batched=True, batch_size=10000)
    print(train, val, test)
    return train, val, test


def load_adapter_model(parameters):
    model_config = AutoConfig.from_pretrained(
        "zhihan1996/DNA_bert_6",
        num_labels=parameters["num_labels"],
        trust_remote_code=True,
    )

    model = BertModelWithHeads.from_pretrained(
        "zhihan1996/DNA_bert_6", config=model_config, trust_remote_code=True
    )
    print("model config: ", model_config)

    return model_config, model


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
    model.add_adapter(parameters["task_name"], config=adapter_config)
    # Add a matching classification head
    # should have the same name as adapter
    model.add_classification_head(
        parameters["task_name"], num_labels=parameters["num_labels"], overwrite_ok=True
    )  # parameters['num_labels'])
    # Activate the adapter and the head
    model.train_adapter(parameters["task_name"])
    # print(model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"number of trainable parameters: {num_trainable_params / 1000000}, \ntotal number of parameters: {model.num_parameters() / 1000000}"
    )

    return model


def add_trained_adapter(model, adapter_name, target_task, with_head=False):
    print(f"adding_{adapter_name}")
    model.load_adapter(
        f"./adapters/DNABERT_{adapter_name}_adapter/{adapter_name}",
        load_as=f"{adapter_name}",
        with_head=with_head,
    )

    print(f"added {adapter_name} adapter")
    return model


def add_and_fuse(model, adapters_list, target_task):
    adapter_names = []
    for adapter in adapters_list:
        model = add_trained_adapter(model, adapter, target_task, with_head=False)
        adapter_names.append(adapter)
    #     model.add_adapter_fusion(
    #                          './adapters/DNABERT_human_0_tf_100_adapter/human_0_tf_100',
    #                          # "human_1_tf_100",
    #                          './adapters/DNABERT_human_2_tf_100_adapter/human_2_tf_100',
    #                          './adapters/DNABERT_human_3_tf_100_adapter/human_3_tf_100',
    #                          './adapters/DNABERT_human_4_tf_100_adapter/human_4_tf_100')
    print(adapters_list)
    model.add_adapter_fusion(adapters_list, "dynamic", {"regularization": True})
    adapter_setup = []
    adapter_setup.append(adapters_list)
    print(adapter_setup)

    print(model.config)
    model.set_active_adapters(adapter_setup)
    # Add a classification head for our target task
    if target_task == "human_reconstructed_splicing_400_adapter":
        num_labels = (3,)
    else:
        num_labels = 2
    model.add_classification_head(target_task, num_labels=num_labels)
    # Unfreeze and activate fusion setup
    # train_adapter_fusion() does two things: It freezes all weights of the model (including adapters!)
    # except for the fusion layer and classification head.
    # It also activates the given adapter setup to be used in very forward pass.
    #     adapter_setup = Fuse(
    #                           './adapters/DNABERT_human_0_tf_100_adapter/human_0_tf_100',
    #                          # "human_1_tf_100",
    #                          './adapters/DNABERT_human_2_tf_100_adapter/human_2_tf_100',
    #                          './adapters/DNABERT_human_3_tf_100_adapter/human_3_tf_100',
    #                          './adapters/DNABERT_human_4_tf_100_adapter/human_4_tf_100')
    model.train_adapter_fusion(adapter_setup)
    print("fusion activated")
    return model


def compute_metrics(val_pred):
    logits, labels = val_pred
    predictions = np.argmax(logits, axis=-1)

    #     if not parameters or parameters['num_labels'] == 2:
    #         average = 'binary'

    #     else:
    #         average = 'weighted'  # for multiclass classification

    accuracy = eval.load("accuracy")  # , average='macro'
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


def save_all_fusion(
    run,
    trainer,
    model,
    test,
    fusion_name,
    save_model=False,
    save_adapter=True,
    save_fusion=True,
    is_sweep=False,
):
    # !!! check for project name to be taken from parameters if uncomment
    if save_model:
        trainer.save_model(f"./models/{model_name}")

    if save_adapter:
        model.save_all_adapters(f"./adapters_for_fusion/{model_name}")
        artifact = wandb.Artifact(
            f"adapters_for_fusion_{model_name}",
            type="adapters_set",
            description=f"DNABERT_adapters_for_fusion_{fusion_name}",
        )
        artifact.add_dir(f"./adapters_for_fusion_/{model_name}")
        run.log_artifact(
            artifact, aliases=[f"{dataset_name}_{fusion_name}_adapters_set"]
        )

    if save_fusion:
        model.save_adapter_fusion(f"./adapters_fusion/{model_name}", fusion_name)
        artifact = wandb.Artifact(
            f"adapters_fusion_{model_name}",
            type="adapters_fusion",
            description=f"DNABERT_{fusion_name}_fusion_tuned_on_GUE_{dataset_name}",
        )
        artifact.add_dir(f"./adapters_fusion/{model_name}")
        run.log_artifact(artifact, aliases=[f"{dataset_name}_{fusion_name}_fusion"])

    # make predictions on test and compute metrics
    predictions = trainer.predict(test)
    print(predictions[2])

    #     # save metrics and predictions on disk
    #     # predictions

    df = pd.DataFrame(data=predictions[0])
    df["label"] = predictions[1]
    probs = torch.nn.functional.softmax(torch.from_numpy(predictions[0]), dim=-1)
    df["probability_0"] = probs[:, 0]
    df["probability_1"] = probs[:, 1]
    df["prediction"] = np.where(df["probability_0"] >= 0.5, 0, 1)
    # wand table requires column names not to be integers
    df.rename(columns={1: "1", 0: "0"}, inplace=True)
    if not is_sweep:
        df.to_csv(f"./results/{model_name}_on_{dataset_name}.csv")

    #     # metrics
    #     metrics_file = "./results/" + model_name + "_metric.txt"
    #     with open(metrics_file, 'w') as f:
    #         for key, value in predictions[2].items():
    #             f.write(f'{key} : {value}\n')

    # log metrics and predictions on wandb
    # note: don't log preds for sweep
    # !!! REVIEW FOR SWEEP

    #     # predictions

    description = (
        f"{model_name} predictions with probabilities from model after run {run.name}"
    )
    if not is_sweep:
        model_pred = wandb.Artifact(
            f"predictions_of_{model_name}_on_{dataset_name}",
            type="evaluation on test",
            description=notebook_name,
        )
        model_pred.add(wandb.Table(dataframe=df), notebook_name)
        run.log_artifact(model_pred)

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

        # add model name and organism - can't add to summary, only numbers
        run.log({"model_name": model_name, "dataset": dataset_name})

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
    print(f"confusion_matrix: {confusion_matrix}")

    return predictions[2]["test_matthews_correlation"]


def training_script(small=True, log=True, task_parameters={}):
    os.environ["WANDB_LOG_MODEL"] = "end"

    parameters = {
        "project_name": project_name,
        "seed": 0,
        # 'model_name': 'zhihan1996/DNA_bert_6',
        "task_name": dataset_name,
        "batch_size": 16,
        "num_train_epochs": 25,
        "save_total_limit": 5,
        "learning_rate": 0.00016016789158802676,  # trial.suggest_loguniform(name='learning_rate', low=2e-6, high=2e-3),
        "weight_decay": 0.010123080803193714,  # trial.suggest_loguniform('weight_decay', 2e-3, 2e-2),
        "warmup_ratio": 0.1,  # trial.suggest_discrete_uniform('warmup_ratio', 0.1, 0.2, 0.05),
        "metric_for_best_model": "matthews_correlation",
        "early_stopping_patience": 4,
        #         'gradient_accumulation_steps' : trial.suggest_categorical(
        #             'gradient_accumulation_steps', [1, 2, 4, 8]
        #         ),
        # add adapter_parameters
        #'reduction_factor' : 16,
        #         'reduction_factor' : trial.suggest_categorical(
        #             'reduction_factor', [2, 4, 16]
        #         ),
        #'non_linearity' : 'swish',  # trial.suggest_categorical('non_linearity', ['swish', 'gelu', 'relu']),
        #'original_ln_before' : False,  # trial.suggest_categorical('original_ln_after', choices=[False, True]),
        #'original_ln_after' : True,
        #         #phm
        #'phm_layer': False,
        #         'phm_layer' : trial.suggest_categorical(name='phm_layer', choices=[False, True]), 'phm_dim' : 4,
        #         'factorized_phm_W' : True, 'shared_W_phm' : False, 'shared_phm_rule' : True,
        #         'factorized_phm_rule' : False, 'phm_c_init' : 'normal', 'phm_init_range' : 0.0001,
        #         'learn_phm' : True, 'hypercomplex_nonlinearity' : 'glorot-uniform',
        #         'phm_rank' : 1, 'phm_bias' : True
    }
    #     for k, v in task_parameters:
    #         parameters[k] = v

    parameters["num_labels"] = task_parameters["num_labels"]
    if int(task_parameters["length"]) > 300:
        parameters["batch_size"] = parameters["batch_size"] // 2

    # first is first
    set_seed(parameters["seed"])

    # init wandb run

    run = wandb.init(
        project=project_name,
        job_type="fusion_training",
        name=f"{model_name}",
        config=parameters,
        save_code=True,
        notes=notebook_name,
        group=group,
        tags=[model_name, organism],
    )
    print(run.name)
    print(notebook_name)  # again, for wandb log file
    wandb.config.update(parameters)
    wandb.log(task_parameters)
    print("task_parameters", task_parameters)  # This way I see them in wandb log
    print("parameters", parameters)  # This way I see them in wandb log

    dataset_art = run.use_artifact(dataset_art_name, type="6mers")

    model_config, model = load_adapter_model(parameters)
    # add adapter and head
    model = add_and_fuse(
        model, task_parameters["adapters_list"], task_parameters["dataset_name"]
    )

    # load and tokenize data
    train_6mers, val_6mers, test_6mers = load_data(small=False)
    train, val, test = tokenize_data(train_6mers, val_6mers, test_6mers)

    # define metrics (necessary for logging)  # or not necessary here :D
    #     accuracy = eval.load("accuracy")
    #     precision = eval.load("precision")
    #     recall = eval.load("recall")
    #     f1 = eval.load("f1")
    #     mcc = eval.load("matthews_correlation")

    num_epochs = parameters["num_train_epochs"]

    training_args = TrainingArguments(
        report_to="wandb",
        output_dir=f"./{model_name}/{run.name}",
        num_train_epochs=num_epochs,
        # save_steps=5921,
        save_strategy="steps",
        save_steps=256,
        save_total_limit=parameters["save_total_limit"],
        evaluation_strategy="steps",
        eval_steps=256,
        load_best_model_at_end=True,
        weight_decay=parameters["weight_decay"],
        learning_rate=parameters["learning_rate"],
        per_device_eval_batch_size=parameters["batch_size"],
        per_device_train_batch_size=parameters["batch_size"],
        warmup_ratio=parameters["warmup_ratio"],
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        label_names=[
            "labels"
        ],  # see https://github.com/huggingface/transformers/issues/22885
        metric_for_best_model=parameters["metric_for_best_model"],
        # gradient_accumulation_steps = parameters['gradient_accumulation_steps']
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

    score = save_all(
        run,
        trainer,
        model,
        test,
        save_model=task_parameters["save_model"],
        save_adapter=task_parameters["save_adapter"],
        is_sweep=task_parameters["is_sweep"],
    )
    print(score)
    run.finish()


# Set up and parameters
if __name__ == "__main__":
    os.environ["WANDB_LOG_MODEL"] = "end"
    fusion_name = "GUE_human_tfs"
    adapters_list = fusion_tasks.GUE_human_tfs

    tasks_list = fusion_tasks.min_human_tfs_set

    for dataset_name in tasks_list:
        task_parameters = {}

        # create task-specific parameters

        organism, feature, task, length = dataset_name.split("_")
        print(organism, task, feature, length)

        #         #mkdirs
        #         if not os.path.exists(f"./train_adapter_{task}"):
        #             os.mkdir(f"./train_adapter_{task}")
        #         if not os.path.exists(f"./train_adapter_{task}/resluts"):
        #             os.mkdir(f"./train_adapter_{task}/resluts")

        # dataset_name = f"{organism}_{feature}_{task}_{length}"
        project_name = f"Fusions"
        model_name = f"DNABERT_fusion_{fusion_name}_on_{dataset_name}"
        notebook_name = f"fusion_{fusion_name}_on_{dataset_name}"  # just convenience
        group = f"{organism}_{task}"
        tag = model_name

        # paths to data
        data_path = f"../datasets/GUE/{organism}_{task}_{length}/{dataset_name}"
        split_path = data_path + "/train-val-test_split/"
        k_mers_path = split_path + "6_mers/"
        # path to model if local
        # model_path = "../results/DNABERT_prom_300_f_2"

        dataset_art_name = (
            f"logvinata/upload_GUE_datasets/{dataset_name}_splitted_6_mers:latest"
        )
        # adapter_art_name = f'logvinata/train_on_GUE/adapter_DNABERT_{adapter_name}_adapter:latest'
        task_parameters["dataset_name"] = dataset_name
        task_parameters["fusion_name"] = fusion_name
        task_parameters["adapters_list"] = adapters_list
        task_parameters["organism"] = organism
        task_parameters["feature"] = feature
        task_parameters["task"] = task
        task_parameters["length"] = length
        task_parameters["project_name"] = project_name
        task_parameters["model_name"] = model_name
        task_parameters["notebook_name"] = notebook_name
        task_parameters["group"] = group
        task_parameters["tag"] = tag

        task_parameters["data_path"] = data_path
        task_parameters["split_path"] = split_path
        task_parameters["k_mers_path"] = k_mers_path
        task_parameters["dataset_art_name"] = dataset_art_name

        # important, subject to change
        task_parameters["save_model"] = False
        task_parameters["save_adapter"] = True
        task_parameters["save_fusion"] = True
        task_parameters["is_sweep"] = False
        if dataset_name == "human_reconstructed_splicing_400":
            task_parameters["num_labels"] = 3
        else:
            task_parameters["num_labels"] = 2

        # print("task_parameters", task_parameters) #This way I see them in wandb log

        training_script(small=True, log=True, task_parameters=task_parameters)
