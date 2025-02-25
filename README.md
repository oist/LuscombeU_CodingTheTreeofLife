# LuscombeU_CodingTheTreeofLife
Clade-specific patterns of scrambling​(Problem Statement and Description):
Bacteria show cross-like patterns of scrambling because they tend to preserve codirectionality between replication and transcription.
So,
   • Can we train an AI model to recognise this pattern:
                -despite circular permutations?
            and -regardless of the extent of scrambling?
   • Can we train an AI model to quantify the extent of scrambling, taking this pattern into consideration?
Code Explanation:
This code is aimed at solving the problem of identifying and quantifying the extent of scrambling in bacteria through AI. It uses a deep learning approach, particularly a hybrid model involving Convolutional Neural Networks (CNN) and structured features from YAML files, to achieve this goal.

Break down of the various steps:

1. Image Preprocessing:
load_image function loads each image, resizes it to a consistent size (224x224 pixels), and normalizes the pixel values between 0 and 1.
This normalization is essential for training deep learning models, as raw pixel values might cause training instability.
2. Feature Extraction from YAML Files:
load_yaml_features loads data from YAML files associated with each image, which contain features like alignment_score, num_scrambles (the target variable), and species.
These features are numeric and are crucial for training the model. For example, the num_scrambles will be the target variable (how much scrambling has occurred).
3. Dataset Loading:
load_data function loads both image and YAML metadata. It checks that the corresponding YAML file exists for each image, appends the image data, and extracts the required features (alignment score, number of scrambles) to form the dataset.
The images and features are prepared for model training, ensuring that each image has corresponding numeric data (alignment score, scramble count).
4. Model Creation:
build_model creates the AI model using a combination of a CNN (ResNet50 pre-trained on ImageNet) for image feature extraction and a simple dense neural network for structured data from the YAML files.
The model consists of two parts:
CNN processes image data and extracts features related to bacterial scrambling.
Dense Network processes the numeric features from the YAML files (alignment scores, scramble counts), allowing the model to learn relationships between these and the target (num_scrambles).
The model is compiled with the Adam optimizer and trained using the mean squared error loss function, appropriate for regression tasks (predicting continuous values like scramble count).
5. Hyperparameter Tuning:
kt.Hyperband is used for hyperparameter optimization, searching for the best combination of model parameters (e.g., number of units in the dense layers and learning rate) to improve model performance.
6. Training:
train_model loads the dataset, scales the YAML features using StandardScaler to normalize them, and splits the data into training and test sets.
The model is then trained using the prepared images and features, and the results (loss and mean absolute error) are printed.
Predictions are made on the test set, and the results are visualized in a scatter plot, comparing actual vs. predicted scramble counts for different bacterial species.
7. Visualization:
visualize_predictions generates a scatter plot comparing the actual scramble counts with the predicted values, where each point is colored by species. This helps evaluate how well the model is predicting scrambling patterns for different species.
Output:
Test Results: The model's performance is evaluated using Mean Absolute Error (MAE), which indicates how close the predicted scramble count is to the actual value.
Visualization: A scatter plot visualizes the relationship between actual and predicted scrambling counts for various bacterial species.
