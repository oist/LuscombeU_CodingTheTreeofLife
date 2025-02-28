import tensorflow as tf
import numpy as np
import os
import yaml
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import albumentations as A  # More advanced augmentation

# Enhanced Augmentation pipeline to simulate circular permutations and scrambling
# Using Albumentations library to apply various augmentations
augmentor = A.Compose([
    A.RandomRotate90(p=0.5),  # Random 90-degree rotations
    A.HorizontalFlip(p=0.5),  # Horizontal flip
    A.VerticalFlip(p=0.5),    # Vertical flip
    A.Affine(scale=(0.9, 1.1)),  # Simulate scaling
    A.OpticalDistortion(distort_limit=0.1, p=0.5),  # Simulate scrambling
    A.GridDistortion(p=0.5),  # Add grid-like distortions
])

# Selected YAML feature keys to be extracted for training the model
selected_keys = [
    "index_strandRand_target",
    "index_strandRand_query",
    "index_synteny_target",
    "index_synteny_query",
    "breakpoint_width_target_Median",
    "breakpoint_width_query_Median",
    "aligned_gaps_target_Median",
    "aligned_gaps_query_Median"
]

def load_image(image_path, target_size=(224, 224)):
    # Load image and resize to target size
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize image
    # Apply advanced augmentation
    img_array = augmentor(image=img_array)['image']
    return img_array

def load_yaml_features(yaml_path):
    # Load YAML file and extract selected features
    with open(yaml_path, 'r') as file:
        try:
            data = yaml.load(file, Loader=yaml.FullLoader)
            if isinstance(data, dict):
                # Extract only the selected features from the YAML
                features = { key: data.get(key, 0) for key in selected_keys }
            else:
                features = { key: 0 for key in selected_keys }  # Return 0 if data is invalid
        except Exception:
            features = { key: 0 for key in selected_keys }  # Return 0 in case of any loading error
    return features

def load_data(image_dir, yaml_dir, target_size=(224, 224)):
    # Load images and their corresponding YAML features
    images, yaml_features_list, labels = [], [], []
    
    # For this example, we are using "breakpoint_width_target_Median" as a proxy for scrambling extent
    target_key = "breakpoint_width_target_Median"
    
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(".o2o_plt.png"):
            # Get base name and corresponding YAML file
            base_name = image_filename.replace(".o2o_plt.png", "")
            yaml_filename = base_name + ".yaml"
            image_path = os.path.join(image_dir, image_filename)
            yaml_path = os.path.join(yaml_dir, yaml_filename)
            
            if os.path.exists(yaml_path):
                # Load image and YAML features
                image = load_image(image_path, target_size)
                features = load_yaml_features(yaml_path)
                
                images.append(image)
                # Keep features in a fixed order as defined in selected_keys
                yaml_features_list.append([features[key] for key in selected_keys])
                labels.append(features.get(target_key, 0))  # Using breakpoint width median as target
                
    return np.array(images), np.array(yaml_features_list), np.array(labels)

# Refined cross-pattern visibility heuristic
def compute_cross_visibility(feature_vector):
    # feature_vector order: [strandRand_target, strandRand_query, synteny_target, synteny_query, ...]
    strand_rand = (feature_vector[0] + feature_vector[1]) / 2.0  # Average of strand randomization
    synteny = (feature_vector[2] + feature_vector[3]) / 2.0  # Average of synteny
    # Adjusted heuristic: emphasize synteny to improve pattern visibility
    cross_visibility = (1 - strand_rand) * synteny**2
    return cross_visibility

def create_model(input_image_shape=(224, 224, 3), input_yaml_shape=(None,)):
    # Create a multi-input model with CNN and Dense layers
    inputs_image = layers.Input(shape=input_image_shape)
    cnn_base = tf.keras.applications.ResNet50(include_top=False, input_tensor=inputs_image)
    cnn_out = layers.GlobalAveragePooling2D()(cnn_base.output)
    
    inputs_yaml = layers.Input(shape=input_yaml_shape)
    x_yaml = layers.Dense(64, activation='relu')(inputs_yaml)
    x_yaml = layers.Dense(32, activation='relu')(x_yaml)
    
    combined = layers.concatenate([cnn_out, x_yaml])
    
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)  # Add dropout to reduce overfitting
    x = layers.Dense(256, activation='relu')(x)
    
    # Regression output (for scrambling extent)
    regression_output = layers.Dense(1, name='scrambling_regression')(x)
    # Classification output (for scrambling categories)
    classification_output = layers.Dense(3, activation='softmax', name='scrambling_classification')(x) 
    
    model = models.Model(inputs=[inputs_image, inputs_yaml], outputs=[regression_output, classification_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'scrambling_regression': 'mse', 'scrambling_classification': 'categorical_crossentropy'},
        loss_weights={'scrambling_regression': 1.0, 'scrambling_classification': 0.5},
        metrics={'scrambling_regression': 'mae', 'scrambling_classification': 'accuracy'}
    )
    return model

def train_model(image_dir, yaml_dir, batch_size=32, epochs=10, target_size=(224, 224)):
    # Load data and split into training and testing sets
    X_images, X_yaml, y = load_data(image_dir, yaml_dir, target_size)
    if len(X_images) == 0:
        print("No data loaded. Check file paths.")
        return None
    
    # Split data into train and test sets
    X_train_img, X_test_img, X_train_yaml, X_test_yaml, y_train, y_test = train_test_split(
        X_images, X_yaml, y, test_size=0.2, random_state=42, stratify=np.digitize(y, bins=np.percentile(y, [33, 66])))
    
    # Standardize YAML features
    scaler = StandardScaler()
    X_train_yaml = scaler.fit_transform(X_train_yaml)
    X_test_yaml = scaler.transform(X_test_yaml)
    
    # Create bins for classification target based on training data distribution
    bins = np.percentile(y_train, [33, 66])
    y_train_class = np.digitize(y_train, bins)
    y_test_class = np.digitize(y_test, bins)
    
    # Convert to categorical labels for classification task
    y_train_class = tf.keras.utils.to_categorical(y_train_class, num_classes=3)
    y_test_class = tf.keras.utils.to_categorical(y_test_class, num_classes=3)
    
    # Compute sample weights for regression based on visibility heuristic
    alpha = 1.0  # Hyperparameter to adjust weight influence
    train_weights = np.array([1.0 + alpha * compute_cross_visibility(x) for x in X_train_yaml])
    test_weights = np.array([1.0 + alpha * compute_cross_visibility(x) for x in X_test_yaml])
    
    model = create_model(input_image_shape=(target_size[0], target_size[1], 3), input_yaml_shape=(X_yaml.shape[1],))
    
    # Check shapes of sample weights and labels
    print("Train weights shape:", train_weights.shape)
    print("Test weights shape:", test_weights.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    model.fit(
        [X_train_img, X_train_yaml],
        {'scrambling_regression': y_train, 'scrambling_classification': y_train_class},
        sample_weight={'scrambling_regression': train_weights, 'scrambling_classification': np.ones(len(y_train))},  # Dictionary for sample_weight
        batch_size=batch_size, epochs=epochs,
        validation_data=([X_test_img, X_test_yaml],
                         {'scrambling_regression': y_test, 'scrambling_classification': y_test_class},
                         {'scrambling_regression': test_weights, 'scrambling_classification': np.ones(len(y_test))})  # Dictionary for validation
    )
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(
        [X_test_img, X_test_yaml],
        {'scrambling_regression': y_test, 'scrambling_classification': y_test_class},
        sample_weight={'scrambling_regression': test_weights, 'scrambling_classification': np.ones(len(y_test))}  # Dictionary for evaluation
    )
    print(f"Test loss: {test_loss}")
    
    return model

# File paths (update these as needed)
image_dir = r"C:\Users\admin\Desktop\CODE\dot_plots"
yaml_dir = r"C:\Users\admin\Desktop\CODE\YAML files"

# Train the model
model = train_model(image_dir, yaml_dir, batch_size=32, epochs=10, target_size=(224, 224))
