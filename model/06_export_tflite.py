import os
import argparse
import tensorflow as tf

def export_tflite(model_path, data_dir, output_path, img_size):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Preparing representative dataset for INT8 quantization...")
    # We need a small representative dataset to calibrate the quantization
    train_dir = os.path.join(data_dir, "Train")
    if not os.path.exists(train_dir):
         print(f"Warning: Train dir {train_dir} not found. Cannot perform INT8 quantization without representative data.")
         rep_ds = None
    else:
        rep_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels=None,
            image_size=(img_size, img_size),
            batch_size=1,
            shuffle=True
        ).take(100) # Take 100 samples for calibration

    def representative_data_gen():
        for input_value in rep_ds:
            yield [tf.cast(input_value, tf.float32)]

    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if rep_ds is not None:
        converter.representative_dataset = representative_data_gen
        # Restrict to INT8 operations
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set input and output tensors to uint8 or int8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model with INT8 quantization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to TFLite format with INT8 quantization.")
    parser.add_argument("--model_path", type=str, default="./saved_models/model_final.keras", help="Path to the saved Keras model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where dataset is located (needed for calibration)")
    parser.add_argument("--output_path", type=str, default="./tflite_model/model_quant.tflite", help="Path to save the TFLite model")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    args = parser.parse_args()
    
    export_tflite(args.model_path, args.data_dir, args.output_path, args.img_size)
