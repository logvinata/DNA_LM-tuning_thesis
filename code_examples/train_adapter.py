#!/usr/bin/env python
# coding: utf-8

# ## Drive, paths and config

# Import

import torch, sys, os

import random

import wandb

import transformers

import numpy as np

import sklearn

from sklearn import metrics

import evaluate as eval

import csv

import pandas as pd

import datasets

from datasets import load_dataset

from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForSequenceClassification,
    BertModelWithHeads, 
    EarlyStoppingCallback, 
    TrainingArguments, 
    Trainer,
    AdapterConfig
)

    
#util functions
# from train_utils import (
#     set_seed,
#     load_data,
#     tokenization,
#     load_adapter_model,
#     add_adapter,
#     compute_metrics,
#     save_all,
# )



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
    train_6mers = load_dataset('csv',
                         data_files=k_mers_path + f"{dataset_name}_train_6mers.csv",
                         delimiter=",")['train']
    val_6mers = load_dataset('csv',
                         data_files=k_mers_path + f"{dataset_name}_val_6mers.csv",
                         delimiter=",")['train']

    test_6mers = load_dataset('csv',
                         data_files=k_mers_path + f"{dataset_name}_test_6mers.csv",
                         delimiter=",")['train']
    if small:
#         small_train_6mers = train_6mers.shuffle(seed=42).select(range(2000))
#         small_val_6mers = val_6mers.shuffle(seed=42).select(range(200))
        small_test_6mers = test_6mers.shuffle(seed=42).select(range(200))
        return small_test_6mers, # small_train_6mers, small_val_6mers, 

    return train_6mers, val_6mers, test_6mers


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


def load_adapter_model():
    
    model_config = AutoConfig.from_pretrained("zhihan1996/DNA_bert_6", num_labels=2, trust_remote_code=True)

    model = BertModelWithHeads.from_pretrained(
         "zhihan1996/DNA_bert_6", config=model_config, trust_remote_code=True
        )
    print("model config: ", model_config)
    
    return model_config, model


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
    model.add_adapter(parameters['task_name'], config=adapter_config)
    # Add a matching classification head
    #should have the same name as adapter
    model.add_classification_head(parameters['task_name'], num_labels=2, overwrite_ok=True) #parameters['num_labels'])
    # Activate the adapter and the head
    model.train_adapter(parameters['task_name'])
    # print(model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable parameters: {num_trainable_params / 1000000}, \ntotal number of parameters: {model.num_parameters() / 1000000}")
    
    return model

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


def save_all(run, trainer, test, save_model=False, save_adapter=False, is_sweep=False):

    # !!! check for project name to be taken from parameters if uncomment
    if save_model:
        trainer.save_model(f"./models/{model_name}")
        
    if save_adapter:
        model.save_all_adapters(f"./adapters/{model_name}")


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
    
    description=f"{model_name} predictions with probabilities from model after run {run.name}"
    if not is_sweep:
        model_pred = wandb.Artifact(
            f"predictions_of_{model_name}_on_{dataset_name}",
            type="evaluation on test",
            description=notebook_name
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
        
    #add model name and organism - can't add to summary, only numbers
        run.log({"model_name" : model_name, "dataset" : dataset_name})

    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(df['label'], df['prediction'], labels=[0, 1])
    wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
                        preds=df['prediction'], y_true=df['label'],
                        class_names=[False, True])})
    print(f"confusion_matrix: {confusion_matrix}")



    return predictions[2]["test_matthews_correlation"]



# Set up and parameters
if __name__ == '__main__':
    
    x = [
        # "human_all-core_promoters_300", 
        # "human_all_promoters_300",
        # "human_nontata_promoters_300", 
        "human_tata_promoters_300", "human_tata-core_promoters_300", "human_nontata-core_promoters_300"]
    
    for dataset_name in x:
        organism, feature, task, length = dataset_name.split("_")
        print(organism, task, feature, length)
        # dataset_name = f"{organism}_{feature}_{task}_{length}"
        project_name = f"train_on_GUE"
        model_name = f"DNABERT_{dataset_name}_adapter"
        notebook_name = f"{model_name}_baseline"  # just convenience
        group = f"{organism}_{task}"
        tag = model_name
        
        #paths to data
        data_path = f"../../datasets/GUE/{organism}_{task}_{length}/{dataset_name}"
        split_path = data_path + "/train-val-test_split/"
        k_mers_path = split_path + "6_mers/"
        #path to model if local
        # model_path = "../results/DNABERT_prom_300_f_2"
        
        dataset_art_name = f'logvinata/upload_GUE_datasets/{dataset_name}_splitted_6_mers:latest'
        # model_art_name = 'logvinata/model-registry/DNA_BERT_full_prom_300:v2'
        
        parameters = {
        'project_name' : project_name,
        'seed' : 0,
        # 'model_name': 'zhihan1996/DNA_bert_6',
        'task_name' : dataset_name,
        "batch_size" : 64,
        'num_train_epochs' : 10,
        'save_total_limit' : 7,
        'learning_rate' : 0.00016016789158802676,  # trial.suggest_loguniform(name='learning_rate', low=2e-6, high=2e-3),
        'weight_decay' : 0.010123080803193714,  # trial.suggest_loguniform('weight_decay', 2e-3, 2e-2),
        'warmup_ratio'  : 00.1,  # trial.suggest_discrete_uniform('warmup_ratio', 0.1, 0.2, 0.05),
        'metric_for_best_model' : 'matthews_correlation',
        'early_stopping_patience' : 4,
            
            
#         'gradient_accumulation_steps' : trial.suggest_categorical(
#             'gradient_accumulation_steps', [1, 2, 4, 8]
#         ),
        # add adapter_parameters
        'reduction_factor' : 16,
#         'reduction_factor' : trial.suggest_categorical(
#             'reduction_factor', [2, 4, 16]
#         ),
        'non_linearity' : 'swish',  # trial.suggest_categorical('non_linearity', ['swish', 'gelu', 'relu']),
        'original_ln_before' : False,  # trial.suggest_categorical('original_ln_after', choices=[False, True]),
        'original_ln_after' : True,
#         #phm
        'phm_layer': False,
#         'phm_layer' : trial.suggest_categorical(name='phm_layer', choices=[False, True]), 'phm_dim' : 4,
#         'factorized_phm_W' : True, 'shared_W_phm' : False, 'shared_phm_rule' : True,
#         'factorized_phm_rule' : False, 'phm_c_init' : 'normal', 'phm_init_range' : 0.0001,
#         'learn_phm' : True, 'hypercomplex_nonlinearity' : 'glorot-uniform',
#         'phm_rank' : 1, 'phm_bias' : True
                 }
        
        parameters['save_model'] = True
        parameters['save_adapter'] = True
        parameters['is_sweep'] = False
        parameters['num_lanels'] = 2

        print("parameters", parameters) #This way I see them in wandb log
        set_seed(parameters['seed'])

        # init wandb run
        print(notebook_name)
        run=wandb.init(project=project_name, job_type="adapter_training", name=f"{model_name}", 
                       config=parameters, save_code=True, notes=notebook_name, 
                       group = group, tags = [model_name, organism])

        print(run.name)
        print(notebook_name) # again, for wandb log file
        wandb.config.update(parameters)

        dataset_art = run.use_artifact(dataset_art_name, type='6mers')
        
        model_config, model = load_adapter_model()
        # add adapter and head
        model = add_adapter(model, parameters)
        
        # load and tokenize data
        train_6mers, val_6mers, test_6mers = load_data(small=False)
        train, val, test = tokenize_data(train_6mers, val_6mers, test_6mers)
        
    #         # define metrics
    #     accuracy = eval.load("accuracy")
    #     precision = eval.load("precision")
    #     recall = eval.load("recall")
    #     f1 = eval.load("f1")
    #     mcc = eval.load("matthews_correlation")
    
        num_epochs = parameters['num_train_epochs']

        training_args = TrainingArguments(
            report_to="wandb",
            output_dir=f"./{model_name}/{run.name}",
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
        
        score = save_all(run, trainer, test, 
                         save_model=parameters['save_model'], 
                         save_adapter = parameters['save_adapter'], 
                         is_sweep = parameters['is_sweep'])
        print(score)
        run.finish()














