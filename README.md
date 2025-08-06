# MambaFRZ
Freeze prediction with Mamba-based predictor. Generation of freeze training dataset, and verification of freeze predictor methodology.

## 📁 Project Directory Structure

```text
MambaFRZ/
├── cka/ # Library containing code for computing CKA
├── data/ # Location of data to use when training models on datasets like CIFAR10/CIFAR100
├── frz_predictor/ # Library for MambaFRZ and SmartFRZ freeze predictors and FRZ dataset generation
├── frz_predictor_training_dataset/ # Ignored by git, folder where all the sampled input weights and corresponding layer labels based on CKA values are stored, training dataset for freeze predictors
├── models/ # Models to use when generating data for freeze predictors, example resnet50, vgg11
├── test_fully_trained_model_weights/ # Weights for use in computing CKA and generating freeze prediction data
├── training_configs/ # Location of JSON configuration files for running experiments for generating freeze training datasets
├── training_frz_predictors/ # Folder for the weights generated when training the freeze predictors MambaFRZ and SmartFRZ on the FRZ dataset
├── ComparePredictionsWithCKALabels.ipynb # Compare predictions made for a specific experiment and seed with the computed CKA values in that experiment, and see if freeze predictor predictions align with CKA curves
├── CompareExperimentResults.ipynb # Compare metrics against two different experiments, view each experiment individually then compare with plots
├── CreateTrainingLabelsForFrzDataset.ipynb # Create FRZ training dataset labels using CKA values
├── RandomSeedDataGenerator.py # File to make use of available GPUs and processes to run multiple experiments with different seeds in parallel. Helpful when multiple GPUs are available on a machine
├── README.md # Project overview and setup instructions
├── TrainFrzPredictor.ipynb # Train FRZ predictors using FRZ training datasets
└── TrainingScriptForCKA.py # Main script used to compute CKA during training and sample layer input weights during training, to get sufficient data on the training dynamics of different neural networks
```