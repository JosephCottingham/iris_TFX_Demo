# Iris TFX Demo


## Setup

Run the following to setup your enviroment

- You must have conda installed this may be done via miniconda or anaconda.

```
conda create --name <env> --file requirements.txt
conda activate <env>
```

## Running Project

- Run the following command to execute the project deployment via the TFX pipeline.

```
python3 run.py
```


## Project Overview

This is a basic project untlizing a TFX pipeline to deploy a model design to classify the standard iris dataset. 

Project File Structure:
- Raw data is stored in `data/*`
- `pipeline.py` contains function that generate the pipeline object
- `trainer.py` contains functions that generate the model for training
- `run.py` when excute acts as a entry point to trigger the pipeline to run localy exporting a new model.

TFX pipeline utlizing the following base components:
- csv_component
- statistics_component
- schema_component
- trainer_component
- evaluator_component
- pusher_component