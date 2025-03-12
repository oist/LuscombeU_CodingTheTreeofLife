import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings (if you get any)
import tensorflow as tf
import numpy as np
import yaml
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers  
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The genomic features that the script will extract from YAML file.
selected_keys = [
    "breakpoint_width_target_Median",
    "aligned_gaps_target_Median",
    "aligned_gaps_query_Median",
    "index_synteny_target",
    "index_synteny_query",
    "index_strandRand_target",
    "index_strandRand_query"
]

# Circular shift function
def circular_shift(image, shift_range=(-50, 50)):
    """Performs a circular shift along the x-axis to simulate genome circular permutations."""
    shift = np.random.randint(shift_range[0], shift_range[1])
    return np.roll(image, shift, axis=1)  # Shift horizontally

# Augmentation pipeline
augmentor = A.Compose([
    A.Rotate(limit=360, p=1.0),  # Simulates circular permutations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.8, 1.2)),
    A.OpticalDistortion(distort_limit=0.2, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
])

# Load and preprocess dot plot images
def load_image(image_path, target_size=(224, 224)):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        
        # Apply Albumentations transformations
        img_array = augmentor(image=img_array)['image']
        
        # Convert to uint8 format before circular shift
        img_array = (img_array * 255).astype(np.uint8)
        
        # Apply circular shift
        img_array = circular_shift(img_array)
        
        # Convert back to float32 for TensorFlow
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Load YAML genomic features
def load_yaml_features(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            return {key: data.get(key, 0) if isinstance(data, dict) else 0 for key in selected_keys}
    except Exception as e:
        print(f"Error loading YAML {yaml_path}: {e}")
        return {key: 0 for key in selected_keys}

# Load dataset
def load_data(image_dir, yaml_dir, target_size=(224, 224)):
    images, yaml_features_list, labels = [], [], []
    target_key = "breakpoint_width_target_Median"
    
    print("Loading data...")
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(".o2o_plt.png"):
            base_name = image_filename.replace(".o2o_plt.png", "")
            yaml_filename = base_name + ".yaml"
            image_path = os.path.join(image_dir, image_filename)
            yaml_path = os.path.join(yaml_dir, yaml_filename)
            
            if os.path.exists(yaml_path):
                image = load_image(image_path, target_size)
                features = load_yaml_features(yaml_path)
                if image is not None:
                    images.append(image)
                    yaml_features_list.append([features[key] for key in selected_keys])
                    labels.append(features.get(target_key, 0))
    
    print(f"Loaded {len(images)} images and {len(yaml_features_list)} YAML entries.")
    return np.array(images), np.array(yaml_features_list), np.array(labels)

# Define CNN + MLP model
def create_model(input_image_shape=(224, 224, 3), input_yaml_shape=(None,)):
    inputs_image = layers.Input(shape=input_image_shape)
    cnn_base = tf.keras.applications.ResNet50(include_top=False, input_tensor=inputs_image, weights='imagenet')
    cnn_out = layers.GlobalAveragePooling2D()(cnn_base.output)
    
    inputs_yaml = layers.Input(shape=input_yaml_shape)
    x_yaml = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.1))(inputs_yaml)
    x_yaml = layers.Dropout(0.5)(x_yaml)
    x_yaml = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1))(x_yaml)
    
    combined = layers.concatenate([cnn_out, x_yaml])
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.1))(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1))(x)
    
    regression_output = layers.Dense(1, name='scrambling_regression')(x)
    classification_output = layers.Dense(3, activation='softmax', name='scrambling_classification')(x)
    
    model = models.Model(inputs=[inputs_image, inputs_yaml], outputs=[regression_output, classification_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'scrambling_regression': 'huber', 'scrambling_classification': 'categorical_crossentropy'},  # Huber loss for regression
        loss_weights={'scrambling_regression': 1.0, 'scrambling_classification': 0.5},
        metrics={'scrambling_regression': 'mae', 'scrambling_classification': 'accuracy'}
    )
    return model

# Train model
def train_model(image_dir, yaml_dir, batch_size=16, epochs=10, target_size=(224, 224)):
    X_images, X_yaml, y = load_data(image_dir, yaml_dir, target_size)
    if len(X_images) == 0:
        print("No data loaded. Check file paths.")
        return None
    
    # Split dataset
    X_train_img, X_test_img, X_train_yaml, X_test_yaml, y_train, y_test = train_test_split(
        X_images, X_yaml, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Scale YAML features
    scaler = StandardScaler()
    X_train_yaml = scaler.fit_transform(X_train_yaml)
    X_test_yaml = scaler.transform(X_test_yaml)
    
    # Scale regression targets
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Categorize scrambling levels
    bins = np.percentile(y_train, [33, 66])
    y_train_class = np.digitize(y_train, bins) - 1
    y_test_class = np.digitize(y_test, bins) - 1
    y_train_class = tf.keras.utils.to_categorical(y_train_class, num_classes=3)
    y_test_class = tf.keras.utils.to_categorical(y_test_class, num_classes=3)
    
    # Train the model
    model = create_model(input_image_shape=(target_size[0], target_size[1], 3), input_yaml_shape=(X_yaml.shape[1],))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        [X_train_img, X_train_yaml], 
        {'scrambling_regression': y_train_scaled, 'scrambling_classification': y_train_class},
        batch_size=batch_size, 
        epochs=epochs, 
        validation_split=0.2, 
        callbacks=[early_stopping],
        verbose=2  # Cleaner output (one line per epoch)
    )
    
    return model, scaler, scaler_y, history

# Paths (Please change them as needed.)
image_dir = r"C:\Users\admin\Desktop\CODE\dot_plots"
yaml_dir = r"C:\Users\admin\Desktop\CODE\YAML files"

# Train and save model
model, scaler, scaler_y, history = train_model(image_dir, yaml_dir)
model.save('scrambling_model.keras')

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot regression loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['scrambling_regression_loss'], label='Training Loss')
    plt.plot(history.history['val_scrambling_regression_loss'], label='Validation Loss')
    plt.title('Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot classification accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['scrambling_classification_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_scrambling_classification_accuracy'], label='Validation Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

# Plot training history
plot_history(history)
