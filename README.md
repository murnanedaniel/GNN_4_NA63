# OneTemplate

A *very* simple template to quickly copy across useful directory structure, and a few useful scripts and models.

## Repository Structure
All scripts to run from the commandline are available at the top-level: `preprocess_events.py`, `train_model.py` and `evaluate_model.py`. These scripts call upon config files in the `Config` directory. They use the Pytorch Lightning modules in the `LightningModules` directory. In that directory, `gnn_base.py` contains a Lightning class that handles all training logic. The `Models/gravnet.py` contains our implementation of the `GravNet` class, which can handle both the vanilla GravNet convolution, and our model GravNetNorm.

## Reproduce Paper Results

### 1. Setup

First, install dependencies with conda:
```bash
conda env create -f gpu_environment.yml
conda activate gnn4na
pip install -e .
```
The following instructions assume a GPU is available, and a host device with around 50Gb of RAM. For logging, we use Weights and Biases. If you wish to use this, you should create an account and `pip install wandb`. 

### 2. Download Data

...

### 3. Preprocess Data

Preprocess the data with:
```bash
python preprocess_events.py Configs/preprocess_config.yaml
```
This can take around 20 minutes (it is done on a single thread for now).

### 4. Train Model

Train the model with:
```bash
python Scripts/train_model.py Configs/small_train_norm.yaml
```

This will train the GravNetNorm model on a small subset of the full dataset. This is useful to ensure that the model is training correctly. To train with the full dataset, use the `full_train_norm.yaml` config file.

Additionally, one can compare with the vanilla GravNet model by using the `full_train_vanilla.yaml` config file.

### 5. Evaluate Model

Evaluate the model with:
```bash
python Scripts/evaluate_model.py Configs/small_train_norm.yaml
```
This will run the best model checkpoint (as determined by validation AUC) on the test dataset, and output the test AUC, accuracy and background rejection rate to the console.

## Citation
...
# GNN_4_NA63
