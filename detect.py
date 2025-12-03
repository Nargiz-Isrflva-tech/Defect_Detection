import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import os
import shutil
import sys
import json

# CONFIGURATION
RAW_DATA_DIR = "raw_defect_data"
PREPROCESSED_DIR = "processed_defect_data"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 25
TEST_IMAGE_FILE = "image_009383.png" # The file name for the prediction test

def enhance_surface_defects(image):
    """Applies filters to highlight surface defects (Binary Mask)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Laplacian filter to detect edges/defects
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    # Convert to binary mask
    _, binary = cv2.threshold(lap, 5, 255, cv2.THRESH_BINARY)
    return binary


def preprocess_data(raw_dir, processed_dir):
    """Processes raw images and saves them to the processed directory by class."""
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory '{raw_dir}' not found.")
        sys.exit(1)

    # Clean and create processed directory
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    total_images = 0

    for class_name in os.listdir(raw_dir):
        raw_class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(raw_class_path):
            continue

        processed_class_path = os.path.join(processed_dir, class_name)
        os.makedirs(processed_class_path, exist_ok=True)

        print(f"Processing class: '{class_name}'...")

        for filename in os.listdir(raw_class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raw_img_path = os.path.join(raw_class_path, filename)
                img = cv2.imread(raw_img_path)
                if img is not None:
                    mask = enhance_surface_defects(img)
                    cv2.imwrite(os.path.join(processed_class_path, filename), mask)
                    total_images += 1

    if total_images == 0:
        print("Error: No images found for processing. Check class folders inside raw_defect_data.")
        sys.exit(1)
        
    print(f"\nProcessing finished. Total images: {total_images}.")


def create_cnn_model(input_shape, num_classes):
    """Creates a Sequential CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def prepare_data_generators(data_dir, target_size, batch_size):
    """Creates an image data generator for training."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.0 # Use all data for training due to small dataset size
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale', # Use grayscale since images were preprocessed to binary
        class_mode='categorical',
        subset='training'
    )
    return train_gen


# FUNCTION TO PRINT ALL WEIGHTS (NPY) CONTENT
def read_model_weights(output_dir="model_info"):
    """Prints the full content of all saved .npy weight files."""
    print(f"\nReading and printing full content of all .npy files in '{output_dir}'...")

    # Increase NumPy print threshold to show full matrices
    try:
        np.set_printoptions(threshold=sys.maxsize, linewidth=200) 
    except ValueError:
        np.set_printoptions(threshold=1000000, linewidth=200) 

    for filename in os.listdir(output_dir):
        if filename.endswith(".npy"):
            path = os.path.join(output_dir, filename)
            
            try:
                weight = np.load(path)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            
            print(f"File: {filename} (Shape: {weight.shape})")
            
            
            # Print the entire matrix content
            print(weight)
            
    # Reset NumPy print options to default
    np.set_printoptions(threshold=1000, linewidth=75)
    print("\nSuccessfully printed all .npy file contents.")


def extract_model_info(model, output_dir="model_info"):
    """Extracts and saves model summary, architecture, and weights."""
    os.makedirs(output_dir, exist_ok=True)

    print("\nModel Architecture Summary:")
    model.summary()

    # Save summary, architecture, and individual weights
    # (Code for saving summary.txt, architecture.json, and .npy files remains)
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    with open(os.path.join(output_dir, "architecture.json"), "w") as f:
        f.write(json.dumps(json.loads(model.to_json()), indent=4))

    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:
            for idx, w in enumerate(weights):
                weight_name = f"layer_{i}_{layer.name}_weights_{idx}.npy"
                np.save(os.path.join(output_dir, weight_name), w)

    # Attempt to plot model image
    try:
        plot_model(model, to_file=os.path.join(output_dir, "model.png"), show_shapes=True)
    except:
        print("Warning: Model image could not be created (pydot/qpdf not installed).")

    print(f"\nModel data saved to the '{output_dir}/' folder.")

    # Print the full content of the weights
    read_model_weights(output_dir)


# MODEL TRAINING
def train_model():
    """Trains the CNN model and saves the trained model."""
    train_gen = prepare_data_generators(PREPROCESSED_DIR, IMAGE_SIZE, BATCH_SIZE)
    num_classes = train_gen.num_classes

    input_shape = IMAGE_SIZE + (1,)
    model = create_cnn_model(input_shape, num_classes)
    
    print("\nStarting model training...")
    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    model_path = "defect_classifier_model.h5"
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # Extract model information after training
    extract_model_info(model)


# TEST FUNCTION
def classify_new_image(model_path, image_path, image_size):
    """Loads the trained model and classifies a new image."""
    print(f"\n--- 3. Classification of '{image_path}' ---")
    if not os.path.exists(image_path):
        print(f"Error: Test image '{image_path}' not found. Place it in the current working directory.")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error: Failed to load model. {e}")
        return

    # Read and preprocess the new image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read image '{image_path}'. CV2 could not open the file.")
        return
        
    # Preprocessing (same as training data)
    processed_mask = enhance_surface_defects(img)
    
    # Resizing and Normalization
    resized_mask = cv2.resize(processed_mask, image_size)
    input_data = np.expand_dims(resized_mask, axis=-1) 
    input_data = np.expand_dims(input_data, axis=0) / 255.0 
    
    # Prediction
    prediction = model.predict(input_data)
    
    # Get class names (Assumes alphabetical order: 'quality' then 'unquality')
    class_names = ['quality', 'unquality'] 
    
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    print(f"\n--- RESULT ---")
    print(f"Predicted Class: **{predicted_class.upper()}**")
    print(f"Confidence Level: **{confidence:.2f}%**")


# MAIN EXECUTION FLOW
if __name__ == "__main__":
    print("--- 1. Data Preprocessing ---")
    preprocess_data(RAW_DATA_DIR, PREPROCESSED_DIR)

    print("\n--- 2. CNN Training ---")
    train_model()

    # 3. Test the Model
    classify_new_image("defect_classifier_model.h5", TEST_IMAGE_FILE, IMAGE_SIZE)