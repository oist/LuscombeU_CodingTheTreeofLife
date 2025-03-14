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
from sklearn.metrics import confusion_matrix, classification_report, r2_score

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

# Define bins for label categorization
bins = [0, 10, 20, 30] 

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

# Data Generator
def data_generator(image_dir, yaml_dir, batch_size=32, target_size=(224, 224)):
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".o2o_plt.png")]
    while True:
        for i in range(0, len(image_filenames), batch_size):
            batch_filenames = image_filenames[i:i + batch_size]
            images, yaml_features, labels = [], [], []
            for image_filename in batch_filenames:
                base_name = image_filename.replace(".o2o_plt.png", "")
                yaml_filename = base_name + ".yaml"
                image_path = os.path.join(image_dir, image_filename)
                yaml_path = os.path.join(yaml_dir, yaml_filename)
                
                if os.path.exists(yaml_path):
                    image = load_image(image_path, target_size)
                    features = load_yaml_features(yaml_path)
                    if image is not None:
                        images.append(image)
                        yaml_features.append([features[key] for key in selected_keys])
                        labels.append(features.get("breakpoint_width_target_Median", 0))
            
            if len(images) > 0:
                # Convert labels to categorical using bins
                binned_labels = np.digitize(labels, bins) - 1
                one_hot_labels = tf.keras.utils.to_categorical(binned_labels, num_classes=len(bins) - 1)
                
                # Yield data as tuples, not lists
                yield (
                    (np.array(images, dtype=np.float32), np.array(yaml_features, dtype=np.float32)),
                    (np.array(labels, dtype=np.float32), np.array(one_hot_labels, dtype=np.float32))
                )

# Define CNN + MLP model
def create_model(input_image_shape=(224, 224, 3), input_yaml_shape=(len(selected_keys),)):
    inputs_image = layers.Input(shape=input_image_shape)
    cnn_base = tf.keras.applications.ResNet50(include_top=False, input_tensor=inputs_image, weights='imagenet')
    cnn_out = layers.GlobalAveragePooling2D()(cnn_base.output)
    
    inputs_yaml = layers.Input(shape=input_yaml_shape)
    x_yaml = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1))(inputs_yaml)
    x_yaml = layers.Dropout(0.5)(x_yaml)
    
    combined = layers.concatenate([cnn_out, x_yaml])
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1))(combined)
    x = layers.Dropout(0.5)(x)
    
    regression_output = layers.Dense(1, name='scrambling_regression')(x)
    classification_output = layers.Dense(3, activation='softmax', name='scrambling_classification')(x)
    
    model = models.Model(inputs=[inputs_image, inputs_yaml], outputs=[regression_output, classification_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'scrambling_regression': 'huber', 'scrambling_classification': 'categorical_crossentropy'},
        loss_weights={'scrambling_regression': 1.0, 'scrambling_classification': 0.5},
        metrics={'scrambling_regression': 'mae', 'scrambling_classification': 'accuracy'}
    )
    return model

# Train model
def train_model(image_dir, yaml_dir, batch_size=16, epochs=10, target_size=(224, 224)):
    # Load data to determine the number of samples and split into train/validation
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".o2o_plt.png")]
    train_filenames, val_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)
    
    # Create data generators
    train_generator = data_generator(image_dir, yaml_dir, batch_size=batch_size, target_size=target_size)
    val_generator = data_generator(image_dir, yaml_dir, batch_size=batch_size, target_size=target_size)
    
    # Define output_signature for the generator
    output_signature = (
        (
            tf.TensorSpec(shape=(None, target_size[0], target_size[1], 3), dtype=tf.float32),  # Images
            tf.TensorSpec(shape=(None, len(selected_keys)), dtype=tf.float32)  # YAML features
        ),
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Regression labels
            tf.TensorSpec(shape=(None, len(bins) - 1), dtype=tf.float32)  # Classification labels
        )
    )
    
    # Convert generators to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_generator,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)
    
    # Train the model
    model = create_model(input_image_shape=(target_size[0], target_size[1], 3), input_yaml_shape=(len(selected_keys),))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_dataset,
        steps_per_epoch=len(train_filenames) // batch_size,
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=len(val_filenames) // batch_size,
        callbacks=[early_stopping],
        verbose=2  # Cleaner output (one line per epoch)
    )
    
    return model, history

# Paths (Please change them as needed.)
image_dir = r"C:\Users\admin\Desktop\CODE\dot_plots"
yaml_dir = r"C:\Users\admin\Desktop\CODE\YAML files"

# Train and save model
model, history = train_model(image_dir, yaml_dir)
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