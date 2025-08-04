# MambaFRZ
Freeze prediction with Mamba-based predictor. Generation of freeze training dataset, and verification of freeze predictor methodology.

## 📁 Project Directory Structure

```text
MambaFRZ/
├── test_fully_trained_model_weights/ # Weights for use in computing CKA and generating freeze prediction data
├── frz_predictor/ # Files related to MambaFRZ and SmartFRZ freeze predictors, and dataset generation.
├── models/ # Models to use when generating data for freeze predictors, example resnet50, vgg11, vgg11_bn
└── README.md # Project overview and setup instructions