# MambaFRZ
Freeze prediction with Mamba-based predictor. Generation of freeze training dataset, and verification of freeze predictor methodology.

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
