# LuscombeU_CodingTheTreeofLife
This repository contains a TensorFlow-based deep learning pipeline for recognizing and classifying scrambling patterns in bacterial genome dot plots. The project integrates image processing, feature extraction from YAML files, data augmentation, and multi-task learning with both regression and classification tasks.

Clade-Specific Patterns of Scrambling:
Bacteria exhibit cross-like patterns of scrambling due to their tendency to preserve codirectionality between replication and transcription. This project addresses the following key questions:
*Can we train an AI model to recognize this pattern, despite circular permutations and regardless of the extent of scrambling?
*Can we quantify the extent of scrambling, while considering the preserved pattern?

Classification Objective
The model classifies input dot plots into:
- Linearly scrambled
- Cross-scrambled

Features
- Deep Learning Model: Uses a ResNet50 backbone with additional fully connected layers.
- Multi-Task Learning:

*Regression Task: Predicts scrambling indices.
*Classification Task: Categorizes scrambling severity.
*Binary Classification: Detects cross-like scrambling patterns.

- Data Augmentation: Utilizes Albumentations for random rotations, flips, and affine transformations.

- Custom Data Generator: Loads and preprocesses image and YAML feature data for training.

- Scalable Training: Supports GPU acceleration, with configurable batch size and epochs.

- Logging & Error Handling: Implements structured logging for better debugging and dataset validation.

Dependencies
TensorFlow, NumPy, OpenCV, Albumentations, scikit-learn, Matplotlib, tqdm

⚠Note: Running on a GPU is strongly recommended.
-GPU Runtime: ~30 minutes to 1 hour.
-CPU Runtime: Exceeds 4 hours due to large dataset size
