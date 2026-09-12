import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import numpy as np

def build_datasets(data_dir, img_size, batch_size, validation_split=0.1):
    train_dir = os.path.join(data_dir, "Train")
    
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="both",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='int'
    )
    
    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")
    
    if "Normal" in class_names:
        normal_idx = class_names.index("Normal")
    else:
        # Fallback if case is different or not found
        normal_idx = [i for i, c in enumerate(class_names) if c.lower() == 'normal']
        if normal_idx:
            normal_idx = normal_idx[0]
        else:
            raise ValueError("Could not find 'Normal' class in dataset.")
            
    print(f"Normal class index is {normal_idx}")
    
    # Map to binary labels: 0 for Normal, 1 for Anomaly
    def to_binary(image, label):
        is_anomaly = tf.cast(label != normal_idx, tf.float32)
        # We can also return the multiclass label if needed, but for now we stick to binary for training
        return image, is_anomaly

    # Data augmentation block
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ])

    def augment(image, label):
        return data_augmentation(image), label

    AUTOTUNE = tf.data.AUTOTUNE
    
    # Calculate class weights based on file counts
    num_normal = 0
    num_anomaly = 0
    for cls in class_names:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.exists(cls_dir):
            num_files = len(os.listdir(cls_dir))
            if cls == class_names[normal_idx]:
                num_normal += num_files
            else:
                num_anomaly += num_files
            
    total = num_normal + num_anomaly
    
    if total > 0:
        weight_for_0 = (1 / num_normal) * (total / 2.0) if num_normal > 0 else 1.0
        weight_for_1 = (1 / num_anomaly) * (total / 2.0) if num_anomaly > 0 else 1.0
    else:
        weight_for_0, weight_for_1 = 1.0, 1.0
        
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated class weights: {class_weight}")

    train_ds = train_ds.map(to_binary, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(to_binary, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_weight

def build_model(img_size):
    # MobileNetV3 expects inputs in [0, 255]
    base_model = applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    
    x = applications.mobilenet_v3.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

def train(data_dir, img_size, batch_size, epochs_phase1, epochs_phase2, lr_phase1, lr_phase2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Preparing datasets...")
    train_ds, val_ds, class_weight = build_datasets(data_dir, img_size, batch_size)
    
    print("Building model...")
    model, base_model = build_model(img_size)
    
    # Phase 1: Train top layer
    print("\n--- Phase 1: Training classification head ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase1),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(output_dir, 'model_phase1.keras'), save_best_only=True)
    ]
    
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        class_weight=class_weight,
        callbacks=callbacks_list
    )
    
    # Phase 2: Fine-tuning
    print("\n--- Phase 2: Fine-tuning top layers of base model ---")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Unfreeze the top N layers
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase2),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks_list_ft = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(output_dir, 'model_final.keras'), save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]
    
    # Determine remaining epochs
    total_epochs = epochs_phase1 + epochs_phase2
    
    # If phase 1 stopped early, adjust initial epoch
    initial_epoch = len(history_1.epoch)
    
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weight,
        callbacks=callbacks_list_ft
    )
    
    print(f"Training complete. Best model saved to {os.path.join(output_dir, 'model_final.keras')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the anomaly detection model.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where the dataset is located")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs_phase1", type=int, default=10, help="Epochs for training head")
    parser.add_argument("--epochs_phase2", type=int, default=15, help="Epochs for fine-tuning")
    parser.add_argument("--lr_phase1", type=float, default=1e-3, help="Learning rate for phase 1")
    parser.add_argument("--lr_phase2", type=float, default=1e-5, help="Learning rate for phase 2")
    parser.add_argument("--output_dir", type=str, default="./saved_models", help="Directory to save models")
    args = parser.parse_args()
    
    train(
        args.data_dir, args.img_size, args.batch_size, 
        args.epochs_phase1, args.epochs_phase2, 
        args.lr_phase1, args.lr_phase2, args.output_dir
    )
