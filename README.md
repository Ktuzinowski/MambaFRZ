# MambaFRZ
Freeze prediction with Mamba-based predictor. Generation of freeze training dataset, and verification of freeze predictor methodology.

## Installation Instructions
To get the same packages that were used inside of this project, follow the below instructions inside the directory of the repo.
This will create a new conda environment called **MambaFRZ**. Use this to run all following Jupyter Notebooks and Python scripts.
```
cd MambaFRZ
conda env create -f environment.yml
```

## 📁 Project Directory Structure

```text
📁 MambaFRZ/
├── 📁 cka/ — Library containing code for computing CKA  
├── 📁 frz_predictor/ — MambaFRZ and SmartFRZ predictors, FRZ dataset generation  
├── 📁 models/ — Models (e.g., ResNet50, VGG11) used for data generation  
├── 📁 training_configs/ — JSON config files for experiments
├── 📁 data/ — (Git-ignored) Data for training models on datasets like CIFAR10/CIFAR100  
├── 📁 frz_predictor_training_dataset/ — (Git-ignored) Sampled weights and layer labels for training predictors  
├── 📁 test_fully_trained_model_weights/ — (Git-ignored) Weights used in CKA computation and freeze prediction  
├── 📁 training_frz_predictors/ — (Git-ignored) Trained weights for MambaFRZ and SmartFRZ  
├── 📊 ComparePredictionsWithCKALabels.ipynb — Compare predictions with CKA values and curves  
├── 📊 CompareExperimentResults.ipynb — Compare experiment metrics with visualizations  
├── 📊 CreateTrainingLabelsForFrzDataset.ipynb — Generate labels for FRZ dataset using CKA  
├── 🐍 RandomSeedDataGenerator.py — Parallel experiments using multiple GPUs  
├── 📖 README.md — Project overview and setup instructions  
├── 🔬 TrainFrzPredictor.ipynb — Train MambaFRZ/SmartFRZ predictors on FRZ dataset  
└── 🐍 TrainingScriptForCKA.py — CKA computation and layer weight sampling script  
```
## How to Use **RandomSeedGenerator.py**
1. In the script, there is a variable called **configuration_file**, set this to whatever JSON file you want, located in the folder **training_configs**
* There is resnet50_cifar100, vgg11_bn_cifar100, vgg11_cifar100, and vgg16_cifar100
* Then each folder has 4 different types of JSON files: 
* **fully_trained_reference_model.json** for creating a fully trained reference model for computing CKA with later on
* **similarity_guided_training.json** for Similarity-Guided training to verify CKA is working properly
* **freeze_dataset_generation.json** for generating the Freeze Training Dataset, with layer input weights as input and labels being generated with the CKA values computed
* **use_frz_predictor_with_post_processing.json** use post-processing windows to take a majority decision from the freeze predictor
2. Next the script will make use of all available GPUs and generate a random seed for each one, specify the number of runs each GPU should run with the variable **number_of_seeds_per_gpu**
3. This completes the generation of the freeze training dataset, you can continue to generating the labels for the freeze training dataset
4. Keep in mind this script passes all configurations to **TrainingScriptForCKA.py**, which does the heavy lifting of the training of models and computing of CKA
## How to Use **CreateTrainingLabelsForFrzDataset.ipynb**
1. Specify the name of the training data folder **name_for_training_data_folder**, in which you want to store the layer input weights and the labels with CKA that are generated
2. Specify the configuration file in which you want the training parameters to be loaded in with **configuration_file_location** variable
3. Follow steps and configuration options within CreateTrainingLabelsForFrzDataset.ipynb, to generate labels and look at CKA curves
## How to Use **TrainFrzPredictor.ipynb**
1. Specify the training data folder within a JSON for the type of training dataset folder you want to use, see examples in the **mambafrz** and **smartfrz** directories
2. Inside the Jupyter Notebook, specify the location to this configuration file using the variable in the second cell **configuration_file_location**
3. Please look at examples in the **mambafrz** and **smartfrz** folders for more details about how to use these freeze predictors and train them, you can specify more than one folder for the training dataset to work
## How to Use **ComparePredictionsWithCKALabels.ipynb**
1. Specify a specific experiment name, and specific seed you would like to look at
2. Experiment must have computed CKA prior so that you can visualize the CKA curves and the predictions being made by the freeze predictors
3. Architecture support may be missing, so you may have to adjust
## How to Use **CompareExperimentResults.ipynb**
1. Specify two experiment names inside of the script, with 2 separate seeds
2. Meant to visualize the CKA curves of corresponding layers in the same experiment, and to look at the variance in the CKA curves
3. In addition, you can compare the predictions and TFLOPs being made between the two predictors