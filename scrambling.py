import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable CPU optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import yaml
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIGURATION 
EPOCHS = 20        # Number of training epochs(please increase them as needed, for running it on CPU I reduced it to 20, can be increased to 50 for running on GPU)
BATCH_SIZE = 16     # Batch size (reduce if memory , and/or can be increased to 32, 64 if needed)
PATIENCE = 5        # Early stopping patience    
TARGET_SIZE = (224, 224)
VISUALIZE_EXAMPLES = True
EXAMPLES_TO_SHOW = 5 

# GPU/CPU Conditional Setup
if tf.config.list_physical_devices('GPU'):
    # GPU-Specific Settings
    BATCH_SIZE = 64  # Larger batches better utilize GPU parallelism
    MIXED_PRECISION = True  # Enable for 2-3x speedup on supported GPUs
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable CPU-specific optims
else:
    # CPU-Specific Settings
    BATCH_SIZE = 8   # Smaller batches prevent memory thrashing
    MIXED_PRECISION = False
      # Set thread configuration for CPU optimization
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(2)

# Enable mixed precision if GPU available and configured
if MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

selected_keys = [
    "breakpoint_width_target_Median",
    "aligned_gaps_target_Median",
    "aligned_gaps_query_Median",
    "index_synteny_target",
    "index_synteny_query",
    "index_strandRand_target",
    "index_strandRand_query"
]
# Binning configuration for classification
bins = {
    "breakpoint_width_target_Median": [1, 10, 100, 1000, 10000, 62421],
    "aligned_gaps_target_Median": [0, 50, 100, 150, 200, 250, 292],
    "aligned_gaps_query_Median": [0, 50, 100, 150, 200, 211],
    # No binning needed for these features
    "index_synteny_target": None,
    "index_synteny_query": None,
    "index_strandRand_target": None,
    "index_strandRand_query": None,
}

# DATA PROCESSING
class DotPlotProcessor:
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size
        self.augmentor = self._build_augmentation_pipeline()
        self.feature_scaler = StandardScaler()
        self.reg_scaler = StandardScaler()
        self._fit_scaler = False
        
    def _build_augmentation_pipeline(self):
        # GPU OPTIMIZATION: Can add more intensive augmentations when using GPU
        # CPU OPTIMIZATION: Simplified pipeline for faster processing
        if tf.config.list_physical_devices('GPU'):
            return A.Compose([
                A.Rotate(limit=360, p=1.0),       # Random rotations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(scale=(0.8, 1.2), p=0.5),
            ], p=1.0)
        else:
            return A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
            ], p=0.8)  # Apply only 80% of samples to reduce load
    
    def circular_shift(self, image, shift_range=(-50, 50)):
        shift = np.random.randint(shift_range[0], shift_range[1])
        return tf.roll(image, shift=shift, axis=1)
    
    def load_image(self, image_path):
        """This part Loads and preprocesses an image file"""
        try:
            # GPU OPTIMIZATION: Can use TF-native augmentations on GPU
            # CPU OPTIMIZATION: Albumentations + numpy is faster on CPU
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, self.target_size)
            img = tf.cast(img, tf.float32) / 255.0
            
            if tf.config.list_physical_devices('GPU'):
                # GPU path - use TF ops
                if tf.random.uniform(()) > 0.5:
                    img = tf.image.flip_left_right(img)
                if tf.random.uniform(()) > 0.5:
                    img = tf.image.flip_up_down(img)
                img = self.circular_shift(img)
            else:
                # CPU path - use Albumentations
                img_np = img.numpy()
                augmented = self.augmentor(image=img_np)['image']
                img = tf.convert_to_tensor(augmented)
                img = self.circular_shift(img)
                
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def weak_label_cross_linear(self, image):
        """Generate weak labels for cross/linear patterns using computer vision"""
        try:
            # Convert to grayscale
            gray = tf.image.rgb_to_grayscale(image)
            gray_uint8 = tf.cast(gray * 255, tf.uint8).numpy()
            
            # Edge detection with optimized parameters
            edges = cv2.Canny(gray_uint8, 50, 150, L2gradient=True)
            
            # Line detection with early exit conditions
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                  threshold=50, 
                                  minLineLength=50, 
                                  maxLineGap=10)
            
            if lines is not None:
                lines = lines[:, 0]
                n = len(lines)
                
                # Return 2 for sparse/no patterns if fewer than 5 lines
                if n < 5:  # Changed threshold
                    return 2  # New "other" category
                
                if n > 20:  # Early exit for clearly cross patterns
                    return 1
                
                intersections = 0
                
                # Count line intersections
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1, x2, y2 = lines[i]
                        x3, y3, x4, y4 = lines[j]
                        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if denom != 0:
                            intersections += 1
                            if intersections > 5:  # Early exit
                                return 1
                
                # If many intersections, label as cross-like pattern
                return 1 if intersections > 5 else 0
            
            # Return 2 for no detectable patterns
            return 2  # Changed from 0 to 2
        except Exception as e:
            logger.error(f"Error in weak labeling: {e}")
            return 2  # Changed from 0 to 2

    def safe_float_convert(self, value):
        try:
            if not value or value.lower() in ['na', '.na', 'nan', 'none', 'null']:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def load_yaml_features(self, yaml_path):
        """Optimized YAML loading with direct iteration"""
        try:
            with open(yaml_path, 'r') as file:
                yaml_data = {}
                for line in file:  # Direct iteration is faster than readlines()
                    line = line.strip()
                    if line and line != "|" and ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        yaml_data[key] = self.safe_float_convert(value)
                
                features = np.array([yaml_data.get(key, 0.0) for key in selected_keys], 
                                  dtype=np.float32)
                
                if not self._fit_scaler:
                    # Using partial_fit for memory efficiency
                    self.feature_scaler.partial_fit(features.reshape(1, -1))
                    self._fit_scaler = True
                
                return self.feature_scaler.transform(features.reshape(1, -1))[0]
        except Exception as e:
            logger.error(f"Error loading YAML {yaml_path}: {e}")
            return None

    def scale_regression_targets(self, targets):
        """Scale regression targets using fitted scaler"""
        if not hasattr(self, 'reg_scaler_fitted'):
            # Using partial_fit for memory efficiency
            self.reg_scaler.partial_fit(np.array(targets).reshape(-1, 1))
            self.reg_scaler_fitted = True
        return self.reg_scaler.transform(np.array(targets).reshape(-1, 1))

    def inverse_scale_regression(self, targets):
        """Inverse transform scaled regression values"""
        return self.reg_scaler.inverse_transform(np.array(targets).reshape(-1, 1))

# DATA GENERATOR
class DotPlotDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, yaml_dir, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, 
                 shuffle=True, file_list=None):
        super().__init__()
        self.image_dir = image_dir
        self.yaml_dir = yaml_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.processor = DotPlotProcessor(target_size)
        
        # CPU OPTIMIZATION: Pre-load all YAML features to RAM
        self._preload_features()
        
        # Initialize valid pairs
        self.image_files = file_list if file_list is not None else [
            f for f in os.listdir(image_dir) if f.endswith(".o2o_plt.png")]
        self.valid_pairs = []
        
        for img_file in self.image_files:
            base = img_file.replace(".o2o_plt.png", "")
            yaml_file = f"{base}.yaml"
            if os.path.exists(os.path.join(self.yaml_dir, yaml_file)):
                self.valid_pairs.append((img_file, yaml_file))
        self.on_epoch_end()
    
    def _preload_features(self):
        """CPU OPTIMIZATION: Load all YAML features once to RAM"""
        self.feature_cache = {}
        for f in tqdm(os.listdir(self.yaml_dir), desc="Preloading features"):
            if f.endswith('.yaml'):
                base = f.split('.')[0]
                features = self.processor.load_yaml_features(os.path.join(self.yaml_dir, f))
                if features is not None:
                    self.feature_cache[base] = features
    
    def __len__(self):
        return int(np.ceil(len(self.valid_pairs) / self.batch_size))
    
    def __getitem__(self, index):
        batch_pairs = self.valid_pairs[index*self.batch_size:(index+1)*self.batch_size]
        return self._generate_batch(batch_pairs)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_pairs)
    
    def _generate_batch(self, batch_pairs):
        batch_images = []
        batch_features = []
        batch_regression = []
        batch_classification = []
        batch_patterns = []
        
        for img_file, yaml_file in batch_pairs:
            img_path = os.path.join(self.image_dir, img_file)
            image = self.processor.load_image(img_path)
            
            # Use preloaded features if available
            base_name = yaml_file.split('.')[0]
            features = self.feature_cache.get(base_name)
            
            if image is not None and features is not None:
                try:
                    # Get YAML data from preloaded features
                    yaml_path = os.path.join(self.yaml_dir, yaml_file)
                    with open(yaml_path, 'r') as f:
                        yaml_data = {}
                        for line in f:
                            line = line.strip()
                            if line and line != "|" and ":" in line:
                                key, value = line.split(":", 1)
                                key = key.strip()
                                value = value.strip()
                                yaml_data[key] = self.processor.safe_float_convert(value)
                    
                    reg_target = yaml_data.get("breakpoint_width_target_Median", 0.0)
                    bin_idx = np.digitize(reg_target, bins["breakpoint_width_target_Median"]) - 1
                    cls_target = tf.keras.utils.to_categorical(
                        bin_idx, 
                        num_classes=len(bins["breakpoint_width_target_Median"]) - 1
                    )
                    
                    # CPU OPTIMIZATION: Only compute patterns during final evaluation
                    if tf.config.list_physical_devices('GPU') or not self.shuffle:
                        pattern_label = self.processor.weak_label_cross_linear(image)
                    else:
                        pattern_label = 0  # Dummy during training
                        
                    pattern_target = tf.keras.utils.to_categorical(pattern_label, num_classes=3)
                    
                    batch_images.append(image)
                    batch_features.append(features)
                    batch_regression.append(reg_target)
                    batch_classification.append(cls_target)
                    batch_patterns.append(pattern_target)
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    continue
        
        if not batch_images:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        # Scale regression targets in batch for efficiency
        batch_regression = self.processor.scale_regression_targets(batch_regression)
        
        # Convert to tensors in single operations
        batch_images = tf.stack(batch_images)
        batch_features = tf.convert_to_tensor(batch_features, dtype=tf.float32)
        batch_regression = tf.convert_to_tensor(batch_regression, dtype=tf.float32)
        batch_classification = tf.convert_to_tensor(batch_classification, dtype=tf.float32)
        batch_patterns = tf.convert_to_tensor(batch_patterns, dtype=tf.float32)
        
        return (
            {'input_image': batch_images, 'input_features': batch_features},
            {
                'scrambling_regression': batch_regression,
                'scrambling_classification': batch_classification,
                'pattern_classification': batch_patterns
            }
        )

# MODEL ARCHITECTURE 
def create_optimized_model(input_image_shape=TARGET_SIZE + (3,), input_features_shape=(len(selected_keys),)):
    input_image = layers.Input(shape=input_image_shape, name='input_image')
    
    # GPU OPTIMIZATION: Use ResNet50 when GPU available
    # CPU OPTIMIZATION: Use MobileNetV2 for faster CPU processing
    if tf.config.list_physical_devices('GPU'):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_tensor=input_image,
            weights='imagenet',
            pooling='avg'
        )
    else:
         base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),  # Explicit shape definition
            include_top=False,
            input_tensor=input_image,
            weights='imagenet',
            pooling='avg'
        )
    
    base_model.trainable = True
    
    input_features = layers.Input(shape=input_features_shape, name='input_features')
    x_features = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_features)
    x_features = layers.BatchNormalization()(x_features)
    x_features = layers.Dropout(0.3)(x_features)
    
    combined = layers.concatenate([base_model.output, x_features])
    
    # GPU OPTIMIZATION: Can use larger layers
    # CPU OPTIMIZATION: Use smaller layers
    if tf.config.list_physical_devices('GPU'):
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
    else:
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
    
    regression_output = layers.Dense(1, name='scrambling_regression')(x)
    classification_output = layers.Dense(
        len(bins["breakpoint_width_target_Median"]) - 1,
        activation='softmax',
        name='scrambling_classification'
    )(x)
    pattern_output = layers.Dense(3, activation='softmax', name='pattern_classification')(x)
    
    model = models.Model(
        inputs=[input_image, input_features],
        outputs=[regression_output, classification_output, pattern_output]
    )
    
    # Enhanced optimizer with adjusted learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,  # Reduced frequency of decay
        decay_rate=0.96)    # Slower decay rate
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'scrambling_regression': 'huber',
            'scrambling_classification': 'categorical_crossentropy',
            'pattern_classification': 'categorical_crossentropy'
        },
        loss_weights={
            'scrambling_regression': 1.0,
            'scrambling_classification': 0.7,
            'pattern_classification': 0.5
        },
        metrics={
            'scrambling_regression': ['mae', 'mse'],
            'scrambling_classification': ['accuracy'],
            'pattern_classification': ['accuracy']
        }
    )
    
    return model

# TRAINING
def train_model(image_dir, yaml_dir):
    all_files = [f for f in os.listdir(image_dir) if f.endswith(".o2o_plt.png")]
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    # GPU OPTIMIZATION: Larger validation batch
    val_batch_size = 64 if tf.config.list_physical_devices('GPU') else 8
    
    train_gen = DotPlotDataGenerator(image_dir, yaml_dir, file_list=train_files)
    val_gen = DotPlotDataGenerator(image_dir, yaml_dir, file_list=val_files, 
                                 shuffle=False, batch_size=val_batch_size)
    
    model = create_optimized_model()
    
    class CleanTrainingCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            print("\nTraining started - 1 line per epoch")
            
        def on_epoch_end(self, epoch, logs=None):
            # Inverse scale regression metrics for reporting
            mae = model.processor.inverse_scale_regression([logs['scrambling_regression_mae']])[0][0]
            val_mae = model.processor.inverse_scale_regression([logs['val_scrambling_regression_mae']])[0][0]
            
            print(f"Epoch {epoch+1}/{EPOCHS}: "
                  f"Loss {logs['loss']:.3f}/{logs['val_loss']:.3f} (train/val) | "
                  f"MAE {mae:.1f} | "
                  f"Acc {logs['scrambling_classification_accuracy']:.2%}/{logs['pattern_classification_accuracy']:.2%}")
    
    callbacks = [
        EarlyStopping(patience=PATIENCE, 
                     monitor='val_scrambling_classification_accuracy',
                     mode='max',
                     restore_best_weights=True),
        ModelCheckpoint('best_model.keras', 
                       save_best_only=True, 
                       monitor='val_scrambling_classification_accuracy',
                       mode='max'),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
        CleanTrainingCallback()
    ]
    
    # Disable tqdm progress bars
    from tqdm import tqdm
    from functools import partial
    tqdm.__init__ = partial(tqdm.__init__, disable=True)
    
    print("\nPreparing data...")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )
    
    tqdm.__init__ = partial(tqdm.__init__, disable=False)
    
    # Attach processor to model for later use
    model.processor = train_gen.processor
    
    return model, history

# VISUALIZATION 
def plot_history(history):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['scrambling_regression_mae'], label='Train')
    plt.plot(history.history['val_scrambling_regression_mae'], label='Validation')
    plt.title('Regression MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['pattern_classification_accuracy'], label='Pattern Train')
    plt.plot(history.history['val_pattern_classification_accuracy'], label='Pattern Val')
    plt.plot(history.history['scrambling_classification_accuracy'], label='Scrambling Train')
    plt.plot(history.history['val_scrambling_classification_accuracy'], label='Scrambling Val')
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_examples(image_dir, linear_files, cross_files):
    """Display example plots of both types"""
    print("\nVisualizing example plots...")
    
    def load_and_show(image_path, title):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.show()
    
    # Show linear examples
    print(f"\nLinear Scrambling Examples ({len(linear_files)} total):")
    for img_file in random.sample(linear_files, min(EXAMPLES_TO_SHOW, len(linear_files))):
        load_and_show(os.path.join(image_dir, img_file), "Linear Scrambling")
    
    # Show cross examples
    print(f"\nCross Scrambling Examples ({len(cross_files)} total):")
    for img_file in random.sample(cross_files, min(EXAMPLES_TO_SHOW, len(cross_files))):
        load_and_show(os.path.join(image_dir, img_file), "Cross Scrambling")
    # Show other pattern examples (ADD THIS BLOCK)
    print(f"\nOther Pattern Examples ({len(other_files)} total):")
    for img_file in random.sample(other_files, min(EXAMPLES_TO_SHOW, len(other_files))):
        load_and_show(os.path.join(image_dir, img_file), "Other Pattern")

# CLASSIFICATION 
def classify_plots(model, image_dir, yaml_dir):
    processor = model.processor
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".o2o_plt.png")]
    
    linear_files = []
    cross_files = []
    other_files = []
    
    print("\nClassifying plots...")
    for img_file in tqdm(image_files, desc="Processing plots"):
        try:
            img_path = os.path.join(image_dir, img_file)
            image = processor.load_image(img_path)
            if image is None:
                continue
            
            base_name = img_file.replace(".o2o_plt.png", "")
            yaml_file = f"{base_name}.yaml"
            yaml_path = os.path.join(yaml_dir, yaml_file)
            features = processor.load_yaml_features(yaml_path)
            if features is None:
                continue
            
            img_input = np.expand_dims(image, axis=0)
            feat_input = np.expand_dims(features, axis=0)
            
            _, _, pattern_pred = model.predict([img_input, feat_input], verbose=0)
            pattern_class = np.argmax(pattern_pred[0])
            
            if pattern_class == 0:
                linear_files.append(img_file)
            elif pattern_class == 1:
                cross_files.append(img_file)
            else:  # Handle other patterns
                other_files.append(img_file)
                
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            continue
    
    # Save results
    with open('linear_plots.txt', 'w') as f:
        f.write('\n'.join(linear_files))
    with open('cross_plots.txt', 'w') as f:
        f.write('\n'.join(cross_files))
    with open('other_plots.txt', 'w') as f:  
        f.write('\n'.join(other_files))
    
    # Print summary
    total = len(linear_files) + len(cross_files) + len(other_files)
    print(f"\nClassification Results:")
    print(f"Linear scrambling plots: {len(linear_files)} ({len(linear_files)/total:.1%})")
    print(f"Cross scrambling plots: {len(cross_files)} ({len(cross_files)/total:.1%})")
    print(f"Other patterns: {len(other_files)} ({len(other_files)/total:.1%})")
    print(f"\nResults saved to linear_plots.txt and cross_plots.txt")
    
    # Visualize examples if enabled
    if VISUALIZE_EXAMPLES and total > 0:
        visualize_examples(image_dir, linear_files, cross_files, other_files)

# MAIN EXECUTION 
if __name__ == "__main__":
    # Paths (Please change them as needed.)
    image_dir = r"C:\Users\admin\Desktop\CODE\Halobacteria\dot plots"
    yaml_dir = r"C:\Users\admin\Desktop\CODE\Halobacteria\YAML Files"
    
    try:
        logger.info("Starting training...")
        model, history = train_model(image_dir, yaml_dir)
        model.save('final_model.keras')
        logger.info("Training completed and model saved")
        plot_history(history)
        classify_plots(model, image_dir, yaml_dir)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise