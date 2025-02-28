# LuscombeU_CodingTheTreeofLife
Clade-specific patterns of scrambling​(Problem Statement and Description):
Bacteria show cross-like patterns of scrambling because they tend to preserve codirectionality between replication and transcription.
So,
   • Can we train an AI model to recognise this pattern:
                -despite circular permutations?
            and -regardless of the extent of scrambling?
   • Can we train an AI model to quantify the extent of scrambling, taking this pattern into consideration?
Code Explanation:
1. Library Imports
TensorFlow & Keras: Used for deep learning model building, training, and evaluation.
NumPy: Used for array handling and numerical operations.
os: Provides functions to interact with the file system.
yaml: Parses YAML files, which contain metadata about the data.
scikit-learn: Provides utilities for data preprocessing, splitting datasets, and scaling.
OpenCV: (not used in the provided code) typically used for image processing.
Albumentations: A library for advanced image augmentation, particularly useful for training deep learning models.
2. Augmentation Pipeline
An image augmentation pipeline is created using Albumentations. This is designed to simulate circular permutations and scrambling of images to increase model robustness:
RandomRotate90: Randomly rotates images by 90 degrees.
HorizontalFlip & VerticalFlip: Randomly flips images horizontally or vertically.
Affine: Applies scaling transformations.
OpticalDistortion & GridDistortion: Simulate distortion or scrambling in the images.
3. Feature Selection
A list selected_keys is defined, which contains the names of features from the YAML files that will be used for training. These features relate to genome alignment, synteny, and breakpoint width.
4. Image and YAML Loading Functions
load_image: Loads and resizes images from a specified path. The images are normalized and augmented using the defined pipeline.
load_yaml_features: Loads a YAML file, extracts relevant features based on selected_keys, and handles errors by returning default values.
load_data: Iterates over the images and their corresponding YAML files, loads them, and prepares them for training. It returns images, feature vectors, and labels.
5. Feature Engineering (Cross-Visibility)
The compute_cross_visibility function computes a visibility score based on certain features like strand_rand and synteny, which are averaged and then used to calculate a weighted "cross visibility." This score influences the regression training.
6. Model Architecture
Multi-input Model: The model takes two inputs:
Image Input: A ResNet50 model is used for feature extraction from images, with the top layer removed and global average pooling added.
YAML Input: The YAML features are passed through two fully connected layers.
Combined Inputs: The image features and YAML features are concatenated and passed through additional dense layers. The final output consists of:
Regression Output: A single value representing the extent of scrambling.
Classification Output: A 3-class classification representing categories of scrambling.
7. Model Compilation
The model is compiled with the Adam optimizer and two loss functions:
Mean Squared Error (MSE) for the regression task.
Categorical Crossentropy for the classification task.
8. Data Preprocessing and Model Training
Data Splitting: The data is split into training and testing sets using train_test_split. The YAML features are standardized using StandardScaler.
Classification Target: The target labels (scrambling_regression) are digitized into 3 categories for classification.
Sample Weights: A custom sample weight is calculated based on the cross visibility, which adjusts the training loss based on the feature importance.
Model Training: The model is trained using the data and augmented images, with weights for regression and classification tasks. The model uses both training data and validation data to minimize the loss.
9. Evaluation
After training, the model is evaluated on the test set, and the loss is printed.
10. File Paths
The paths to the image directory (dot_plots) and YAML directory (YAML files) are specified.
-Summary of What the Code Does:
Preprocessing: Loads images and metadata (YAML files), applies augmentations to the images, and scales the YAML features.
Modeling: Builds a deep learning model that combines image data and structured data (from YAML files) for a multi-task learning scenario: a regression task to predict scrambling extent and a classification task to predict scrambling categories.
Training: The model is trained with both image and metadata inputs, using a custom sample weighting mechanism based on feature visibility.
Evaluation: The model's performance is evaluated on a test set
