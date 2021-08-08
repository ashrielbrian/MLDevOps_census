# Salary prediction Random Forest Classifier with a Web API 

A random forest binary classifier of an individual's salary based on census data, with a FastAPI implementation deployed on Heroku.

## Introduction

This repo holds the code to:
1. Clean and process raw census data
2. Train a RFC on a the processed data
3. Saves the model, label binarizer, encoder and datasets to a remote `dvc` repo
4. Deploy the model to Heroku

For further information on the data used and model, please see the [model card](model_card.md).

## Getting started

1. Clone the repo:
```bash
    git clone https://github.com/ashrielbrian/MLDevOps_census.git
```

2. Create a virtual env. Conda is used here, but `venv` would work just the same.

```bash
    conda create --name <env-name> python=3.8
    conda activate <env-name>
```

3. Within the virtual env, and in the local root directory,
```bash
    pip install -r requirements.txt
```

4. Setup `dvc` [as below](#dvc-setup).

5. Train the model. Model training will save the model and data objects in the `artifact/` directory, which is where the web component expects to find the pickle objects to load.

    Configure the model hyperparameters in `model/model_config.yaml`.
```bash
    python model/train_model.py
```

Once the model has been trained and artifacts saved in `artifact/`, be sure to add them with `dvc add <artifact-path>` and `git add` the appropriate dvc output files. Finally, `dvc push` to push the artifacts to the remote repo.

6. Once done, proceed to setup CI using Github Actions and CD [with Heroku](#heroku-deployment).

---
## DVC Setup
This project uses `dvc` for artifact version control. First, install dvc (within your virtual env):

```bash
    pip install dvc==2.5.4
```

The dvc remote location is found in the file `.dvc/config` - amend as necessary. Alternatively, set the default remote:

```bash
    dvc remote add -d <remote-name> s3://bucket/folder
```

If using S3 as a remote repository, then you would have needed to run `pip install dvc[s3]`. However, the extra dependencies of `dvc[s3]` has already been added to the `requirements.txt` as:

- `s3fs==2021.7.0`
- `aiobotocore[boto3]>1.0.1`

So you can ignore the `dvc[s3]` - you will still need to install `dvc` as above, however.

**Important Note**: both Github Actions and Heroku requires the artifacts to be pulled from S3 via `dvc pull`. Be sure to include the AWS credentials and dvc setup for both build environments.

---
## Heroku Deployment

This project uses Github actions (`.github/workflows`) for CI and Heroku for CD.

### Buildpacks required:
1. `heroku/python`
2. `heroku-community/apt`   - required to install apt packages (like `dvc`)

### Files required:
1. `Procfile`       - provides Heroku with the dyno startup commands
2. `Aptfile`        - apt dependencies to install. Here, it's the `dvc` version.
3. `runtime.txt`    - the Python runtime version associated to Heroku's buildpack

### Config Variables required:
1. `AWS_ACCESS_KEY_ID`
2. `AWS_SECRET_ACCESS_KEY`

These env variables are required as the dvc remote is in a S3 bucket.
