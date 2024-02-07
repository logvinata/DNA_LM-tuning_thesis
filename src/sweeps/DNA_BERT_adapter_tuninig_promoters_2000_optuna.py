#!/usr/bin/env python
# coding: utf-8



# ## Install and import

import torch
import sys

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("NO CUDA!!!")
    sys.exit(1)

import wandb
import os
import transformers
import numpy as np
import sklearn
import evaluate as eval
import datasets
from datasets import load_dataset
import random
import csv
import pandas as pd
from sklearn import metrics
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    BertModelWithHeads, 
    EarlyStoppingCallback, 
    TrainingArguments, 
    Trainer,
    AdapterConfig
)


import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import argparse



# ## Utils
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

    
def load_model():
    
    model_config = AutoConfig.from_pretrained("zhihan1996/DNA_bert_6", num_labels=2, trust_remote_code=True)

    model = BertModelWithHeads.from_pretrained(
         "zhihan1996/DNA_bert_6", config=model_config, trust_remote_code=True
        )
    print("model config: ", model_config)
    return model_config, model
    
    
def load_data(small=False):
    train_6mers = load_dataset('csv',
                         data_files=split_path + f"6_mers/hg38_{task_name}_train_6mers.csv",
                         delimiter=",")['train']
    val_6mers = load_dataset('csv',
                         data_files=split_path + f"6_mers/hg38_{task_name}_val_6mers.csv",
                         delimiter=",")['train']

    test_6mers = load_dataset('csv',
                         data_files=split_path + f"6_mers/hg38_{task_name}_test_6mers.csv",
                         delimiter=",")['train']
    if small:
        print("creating small datasets")
        small_train_6mers = train_6mers.shuffle(seed=42).select(range(2000))
        small_val_6mers = val_6mers.shuffle(seed=42).select(range(200))
        small_test_6mers = test_6mers.shuffle(seed=42).select(range(200))
        return small_train_6mers, small_val_6mers, small_test_6mers

    return train_6mers, val_6mers, test_6mers

# ### Add Adapter
def add_adapter(model, parameters):
    # Define an adapter configuration
    adapter_config = AdapterConfig.load(
        "pfeiffer", reduction_factor=parameters['reduction_factor'], 
        non_linearity=parameters['non_linearity'], 
        original_ln_before=parameters['original_ln_before'], 
        original_ln_after=parameters['original_ln_after']
    )
    print(f"adapter_config : {adapter_config}")
    # Add a new adapter
    model.add_adapter("prom_300", config=adapter_config)
    # Add a matching classification head
    #should have the same name as adapter
    model.add_classification_head("prom_300", num_labels=2, overwrite_ok=True)
    # Activate the adapter and the head
    model.train_adapter("prom_300")
    # print(model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable parameters: {num_trainable_params / 1000000}, \ntotal number of parameters: {model.num_parameters() / 1000000}")
    
    return model



# ## Dataset

def tokenization(sequences_batch):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", do_lower_case=False, trust_remote_code=True)
    return tokenizer(sequences_batch["kmer"],
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

    # !!! check for project name to be taken from parameters if uncomment
    # trainer.save_model("./results/DNABERT_" + task_name)

    # make predictions on test and compute metrics
    predictions = trainer.predict(test)
    print(predictions[2])

#     # save metrics and predictions on disk
#     # predictions
    df = pd.DataFrame(data=predictions[0])
    df["label"]=predictions[1]
    probs = torch.nn.functional.softmax(torch.from_numpy(predictions[0]), dim=-1)
    df["probability_0"] = probs[:, 0]
    df["probability_1"] = probs[:, 1]
    df["prediction"] = np.where(df["probability_0"] >= 0.5, 0, 1)
    #wand table requires column names not to be integers
    df.rename(columns = {1 : '1', 0 : '0'}, inplace = True)
    df.to_csv("./results/" + task_name + ".csv")

#     # metrics
#     metrics_file = "./results/" + model_name + "_metric.txt"
#     with open(metrics_file, 'w') as f:
#         for key, value in predictions[2].items():
#             f.write(f'{key} : {value}\n')


    # log metrics and predictions on wandb
    # note: don't log preds for sweep
    # !!! REVIEW FOR SWEEP

#     # predictions
    description=f"{task_name} predictions with probabilities from model after run {run.name}"
    prob_table = wandb.Table(dataframe=df)
    model_pred = wandb.Artifact(
        f"DNABERT_{task_name}_pred",
        type="full tuned model predictions on test",
        description=description
        )
    model_pred.add(wandb.Table(dataframe=df), description)
    run.log_artifact(model_pred, aliases=['adapter tuning predictions', 'baseline'])

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
    confusion_matrix = metrics.confusion_matrix(df['label'], df['prediction'], labels=[0, 1])
    wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
                        preds=df['prediction'], y_true=df['label'],
                        class_names=[False, True])})



    return predictions[2]["test_matthews_correlation"]

def training_script(trial=optuna.trial.Trial, small=True, log = True):
    os.environ['WANDB_LOG_MODEL'] = 'end'
    # os.environ['WANDB_WATCH'] = 'all'
    # os.environ['WANDB_NOTEBOOK_NAME'] = 'DNA_BERT_full_tuninig_promoters_2000'


    parameters = {
        'project_name' : f"DNA_BERT_sweeps_{task_name}",
        'seed' : 0,
        # 'model_name': 'zhihan1996/DNA_bert_6',
        'task_name' : task_name,
        "batch_size" : 32,
        'num_train_epochs' : 10,
        'save_total_limit' : 7,
        'learning_rate' : trial.suggest_loguniform(name='learning_rate', low=2e-6, high=2e-3),
        'weight_decay' : trial.suggest_loguniform('weight_decay', 2e-3, 2e-2),
        'warmup_ratio'  : trial.suggest_discrete_uniform('warmup_ratio', 0.1, 0.2, 0.05),
        'metric_for_best_model' : 'matthews_correlation',
        'early_stopping_patience' : 3,
#         'gradient_accumulation_steps' : trial.suggest_categorical(
#             'gradient_accumulation_steps', [1, 2, 4, 8]
#         ),
        # add adapter_parameters
        'reduction_factor' : trial.suggest_categorical(
            'reduction_factor', [2, 4, 16]
        ),
        'non_linearity' : trial.suggest_categorical('non_linearity', ['swish', 'gelu', 'relu']),
        'original_ln_before' : trial.suggest_categorical('original_ln_after', choices=[False, True]),
        'original_ln_after' : True,
#         #phm
        'phm_layer': False,
#         'phm_layer' : trial.suggest_categorical(name='phm_layer', choices=[False, True]), 'phm_dim' : 4,
#         'factorized_phm_W' : True, 'shared_W_phm' : False, 'shared_phm_rule' : True,
#         'factorized_phm_rule' : False, 'phm_c_init' : 'normal', 'phm_init_range' : 0.0001,
#         'learn_phm' : True, 'hypercomplex_nonlinearity' : 'glorot-uniform',
#         'phm_rank' : 1, 'phm_bias' : True
                 }

    print("parameters", parameters) #This way I see them in wandb log
    set_seed(parameters['seed'])


    # init wandb run
    if small:
        note = "test run on a micro dataset"
        # model_name = "logging_test"
    else:
        note = "full model " + task_name
        
    
    # run = trial.wandb.run
    run = wandb.init(project=parameters["project_name"], job_type="adapter_train_hyperoptimization", 
                     config=parameters, save_code=True, notes=note, tags=["optuna"], group="DDP")
    print(run.name)
    print(note) # again, for wandb log file
    wandb.config.update(parameters)
    
    # log the data usage
    run.use_artifact(f'logvinata/DNA_BERT_promoters_2000/promoters_2000_splitted_6_mers:v0', type='6mers')
    # logvinata/DNA_BERT_promoters_2000/promoters_2000_splitted_6_mers:v0', type='6mers'

    # load model and tokenizer
    model_config, model = load_model()
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
#     if small:
#         num_epochs = 2
#     else:
#         num_epochs = parameters['num_train_epochs']

    num_epochs = parameters['num_train_epochs']

    training_args = TrainingArguments(
        report_to="wandb",
        output_dir="./results_" + run.name,
        num_train_epochs=num_epochs,
        # save_steps=5921,
        save_strategy="steps",
        save_steps=256,
        save_total_limit=parameters['save_total_limit'],
        evaluation_strategy="steps",
        eval_steps=256,
        load_best_model_at_end=True,
        weight_decay=parameters['weight_decay'],
        learning_rate=parameters['learning_rate'],
        per_device_eval_batch_size=parameters["batch_size"],
        per_device_train_batch_size=parameters["batch_size"],
        warmup_ratio = parameters['warmup_ratio'],
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        label_names=["labels"], # see https://github.com/huggingface/transformers/issues/22885
        metric_for_best_model=parameters['metric_for_best_model'],
        # gradient_accumulation_steps = parameters['gradient_accumulation_steps']

    )

    #add earlystopping
    early_stop = [EarlyStoppingCallback(early_stopping_patience=parameters['early_stopping_patience'])]

    # finally trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        callbacks = early_stop,
    )

    # train
    trainer.train()


    if log:
        score = save_all(run, trainer, test)

    #     # and finally model seems to be not logged automatically
#     model_artifact=wandb.Artifact(task_name, type="model",
#                      description="baseline")
#     model_artifact.add(model)
#     run.log_artifact(model_artifact, aliases=['full tuning', 'baseline', task_name])

    # return model

    run.finish()
    return score

# https://github.com/optuna/optuna-examples/blob/main/wandb/wandb_integration.py
if __name__ == "__main__":
    # ## Drive, paths and config
    os.environ['WANDB_LOG_MODEL'] = 'end'
    parser = argparse.ArgumentParser()
#     parser.add_argument('--test', nargs='?', const=True, type=bool, default=True)
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    small = args.test
    print(small)
    
    wandb_kwargs = {"project": "DNA_BERT_sweeps_promoters_2000", "job_type" : "adapter_train_hyperoptimization", 
                     "group" : "DDP", "save_code" : True,}
   
    # os.environ['WANDB_WATCH'] = 'all'
    
    task_name = "promoters_2000"
    task_type = "promoters"
    data_path = f"../../datasets/{task_type}/{task_name}/"
    split_path = data_path + "train-val-test_split/"
    
    n_trials = 2 if small == True else 14
    
    wandbc = WeightsAndBiasesCallback(metric_name="matthews_correlation", wandb_kwargs=wandb_kwargs)
    
    study = optuna.create_study(direction="maximize")
    
    study.optimize(lambda trial: training_script(trial, small=small, log = True), n_trials=n_trials, callbacks=[wandbc])  
    # without lambda won't get trial argument
    
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
      
    with open("optuna_results.txt", "a") as f:
        f.write(f"Number of finished trials: {len(study.trials)}")
        if small == True:
            f.write("This is a test")
        f.write("Best trial:")
        f.write(f"  Value: {trial.value}")
        f.write(f"  Params: ")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}")
            
     