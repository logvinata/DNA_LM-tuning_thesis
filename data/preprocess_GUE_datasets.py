# read the files
import os

import datasets
import pandas as pd
import wandb
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict

# list of datasets in the same folder
from datasets_list import datasets_list


def return_kmer_simple(seq, K=6):
    """
    This function outputs the K-mers of a sequence
    Taken from DNABERT github repo
    Parameters
    ----------
    seq : str
        A single sequence to be split into K-mers
    K : int, optional
        The length of the K-mers, by default 6
    Returns
    -------
    kmer_seq : str
        A string of K-mers separated by spaces
    """

    kmer_list = []
    for x in range(len(seq) - K + 1):
        kmer_list.append(seq[x : x + K])

    kmer_seq = " ".join(kmer_list)

    return kmer_seq


if __name__ == "__main__":
    for ds in datasets_list:
        organism, feature, task, length = ds.split("_")
        print(organism, task, feature, length)
        dataset_name = f"{organism}_{feature}_{task}_{length}"
        notebook_name = f"wandb_GUE_{dataset_name}_dataset_upload"  # just convenience
        project_name = f"upload_GUE_datasets"

        # dirs for this dataset
        data_path = f"{organism}_{task}_{length}/{dataset_name}/"
        split_path = data_path + "/train-val-test_split/"
        k_mer_path = split_path + "6_mers/"

        #         os.mkdir(f"{k_mer_path}")

        #         # Split
        #         # the split is already there

        train = load_dataset("csv", data_files=f"{split_path}train.csv", delimiter=",")[
            "train"
        ]

        val = load_dataset("csv", data_files=f"{split_path}dev.csv", delimiter=",")[
            "train"
        ]

        test = load_dataset("csv", data_files=f"{split_path}test.csv", delimiter=",")[
            "train"
        ]

        # log to wandb
        print("log initial splitted dataset")
        run = wandb.init(project=project_name, job_type="upload")
        split_data = wandb.Artifact(
            f"{dataset_name}_splitted_data",
            type="splitted_dataset",
            description=f"splitted {dataset_name}_GUE_dataset_25.07.23",
        )

        print(train, val, test)  # let's have it in wandb log file
        names_list = ["train.csv", "dev.csv", "test.csv"]
        for name in names_list:
            # add file
            split_data.add_file(split_path + name)
            print(f"{name} added to artifact")

            temp_df = pd.read_csv(split_path + name)
            # now add table to an artifact
            table = wandb.Table(dataframe=temp_df)
            split_data.add(table, name)
            print(f"table {dataset_name}_{name} added to artifact")
        run.log_artifact(split_data)
        run.finish()

        ## Make 6-mers for DNABERT

        k_mer_train = train.map(
            lambda batch: {
                "kmer": [return_kmer_simple(seq) for seq in batch["sequence"]]
            },
            remove_columns=["sequence"],
            batched=True,
            batch_size=len(train),
        )
        k_mer_val = val.map(
            lambda batch: {
                "kmer": [return_kmer_simple(seq) for seq in batch["sequence"]]
            },
            remove_columns=["sequence"],
            batched=True,
            batch_size=len(val),
        )
        k_mer_test = test.map(
            lambda batch: {
                "kmer": [return_kmer_simple(seq) for seq in batch["sequence"]]
            },
            remove_columns=["sequence"],
            batched=True,
            batch_size=len(test),
        )

        k_mer_train.to_csv(k_mer_path + f"{dataset_name}_train_6mers.csv")
        k_mer_val.to_csv(k_mer_path + f"{dataset_name}_val_6mers.csv")
        k_mer_test.to_csv(k_mer_path + f"{dataset_name}_test_6mers.csv")

        with wandb.init(
            project=project_name,
            job_type="upload"
            # save_code=True, # upload code to wandb
        ) as run:
            artifact = run.use_artifact(
                f"logvinata/{project_name}/{dataset_name}_splitted_data:latest",
                type="splitted_dataset",
            )
            artifact = wandb.Artifact(
                f"{dataset_name}_splitted_6_mers",
                type="6mers",
                description=f"splitted {dataset_name}_dataset_from_GUE_25.07.2023",
            )
            artifact.add_dir(k_mer_path)
            run.log_artifact(artifact, aliases=[f"{dataset_name}_6_mers"])
