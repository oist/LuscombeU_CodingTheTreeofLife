# LuscombeU_CodingTheTreeofLife
This repository contains a TensorFlow-based deep learning pipeline for recognizing and classifying scrambling patterns in bacterial genome dot plots. The project integrates image processing, feature extraction from YAML files, data augmentation, and multi-task learning with regression and classification outputs.

Clade-Specific Patterns of Scrambling: Problem Statement and Description-
Bacteria show cross-like patterns of scrambling because they tend to preserve codirectionality between replication and transcription.
This project addresses the following questions:
-Can we train an AI model to recognize this pattern: Despite circular permutations? or Regardless of the extent of scrambling?
-Can we train an AI model to quantify the extent of scrambling while considering this pattern?
Classification Objective: The model also classifies input dot plots as either linearly scrambled or cross-scrambled.

Features-
*Deep Learning Model: Uses a ResNet50 backbone with additional fully connected layers.
*Multi-Task Learning:
-Regression task to predict scrambling indices.
-Classification task to bin scrambling severity.
-Binary classification for detecting cross-like patterns.
*Data Augmentation: Uses Albumentations for random rotations, flips, and affine transformations.
*Custom Data Generator: Loads and preprocesses image and YAML feature data for training.
*Scalable Training: Supports GPU acceleration with adjustable batch sizes and epochs.
*Logging and Error Handling: Implements logging for better debugging and dataset validation.
