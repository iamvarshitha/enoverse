import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import History
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt 
import json

# --- 1. CONFIGURATION ---
# The path to the main folder containing the class subdirectories.
DATASET_DIR = '.' 

# Image and Model Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10 
VALIDATION_SPLIT = 0.2
MODEL_SAVE_PATH = 'corn_classifier_model.keras'
CLASS_NAMES_PATH = 'class_names.json' 
# CLASS_NAMES is now managed by get_class_names() and not primarily global state

# --- 2. UTILITY FUNCTIONS ---

# New robust function to load class names from the persisted JSON file
@st.cache_resource
def get_class_names():
    """Loads class names from the persisted JSON file, or [] if file is missing."""
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, 'r') as f:
                names = json.load(f)
                return names
        except Exception as e:
            st.error(f"Failed to read class names from JSON: {e}")
            return []
    return []

# Function to capture model summary for Streamlit display
def get_model_summary(model):
    """Captures the model summary text into a string."""
    stringlist = []
    # Use the native Keras function to print to a string list
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)

# --- 3. DATA LOADING AND TRAINING FUNCTIONS ---

@st.cache_resource
def load_and_train_model():
    """Loads data, builds, and trains the CNN model."""
    
    st.info(f"Starting data load from directory: '{os.path.abspath(DATASET_DIR)}'...")
    
    try:
        # Load and split training data (80%)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )

        # Load validation data (20%)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )

        CLASS_NAMES_local = train_ds.class_names # Use a local variable for certainty
        NUM_CLASSES = len(CLASS_NAMES_local)
        st.success(f"Classes found: {CLASS_NAMES_local}")
        st.markdown(f"**Training Images:** {len(train_ds.file_paths)} files")
        st.markdown(f"**Validation Images:** {len(val_ds.file_paths)} files")
        
        if NUM_CLASSES == 0:
             st.error("Found 0 images. Please check that your class folders contain images.")
             st.stop()
        
    except Exception as e:
        st.error(f"Error loading data. Check if your class folders are directly inside the current directory. Error details: {e}")
        st.stop()

    # Preprocessing: Rescaling pixel values from 0-255 to 0-1
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    def prepare_dataset(ds):
        ds = ds.map(lambda x, y: (normalization_layer(x), y))
        ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = prepare_dataset(train_ds)
    val_ds = prepare_dataset(val_ds)
    
    # --- Model Definition with Data Augmentation ---
    data_augmentation = Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name="data_augmentation")

    model = Sequential([
        # 1. Augmentation Layer (applies only during training)
        data_augmentation,
        
        # 2. Convolutional Layers
        Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.4),
        
        # New: Deeper layer
        Conv2D(256, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.4),

        # 3. Classification Head
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.6),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    st.subheader("Model Summary")
    st.code(get_model_summary(model), language='text')
    
    # --- Model Training ---
    st.subheader("Model Training")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=0
    )
    
    # --- Persistence: Save Model and Class Names ---
    model.save(MODEL_SAVE_PATH)
    
    # Save the class names to a JSON file
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(CLASS_NAMES_local, f) # Save the local variable!
        
    st.success(f"Training finished. Model saved to '{MODEL_SAVE_PATH}' and class names to '{CLASS_NAMES_PATH}'.")
    
    return model, history

# --- 4. PREDICTION FUNCTION ---

@st.cache_resource
def load_trained_model():
    """Loads the model from disk."""
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(CLASS_NAMES_PATH):
        try:
            # Check if we can get the class names first
            names = get_class_names()
            if not names:
                st.sidebar.error("Class names file is present but empty. Please retrain.")
                return None
            
            # Load the model
            model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=True)
            st.sidebar.success("Model loaded successfully for prediction!")
            return model
            
        except Exception as e:
            st.sidebar.error(f"Failed to load model or class names: {e}")
            
    st.sidebar.warning("Model or class names file not found. Please click 'Train Model' first.")
    return None

def predict_image(model, image):
    """Predicts the class of a single uploaded image."""
    
    # CRITICAL: Always retrieve the class names from the robust function here
    current_class_names = get_class_names()
    
    if not current_class_names:
        # Re-raise the error specifically from the prediction logic
        raise ValueError("Class names are empty. The model cannot make a labeled prediction.")

    # Convert PIL image to numpy array and resize
    img_array = np.array(image.resize((IMG_HEIGHT, IMG_WIDTH)))
    # Add a batch dimension (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    # Normalize (0-1)
    img_batch = img_batch.astype('float32') / 255.0

    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    # Get the softmax scores (probabilities)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class_index = np.argmax(score)
    predicted_class = current_class_names[predicted_class_index] 
    confidence = np.max(score)

    return predicted_class, confidence, score.numpy()


# --- 5. STREAMLIT APP LAYOUT ---

def plot_history(history: History):
    """Plots training and validation metrics."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax[0].plot(history.history['accuracy'], label='Training Accuracy', color='#10B981')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#EF4444')
    ax[0].set_title('Accuracy over Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--')

    # Loss Plot
    ax[1].plot(history.history['loss'], label='Training Loss', color='#10B981')
    ax[1].plot(history.history['val_loss'], label='Validation Loss', color='#EF4444')
    ax[1].set_title('Loss over Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    ax[1].grid(axis='y', linestyle='--')
    
    st.pyplot(fig)


def main_app():
    st.set_page_config(layout="wide", page_title="Corn Image Classifier")
    
    st.title("🌽 Corn Variety Image Classifier")
    st.caption("A Streamlit application for training and predicting corn images using a CNN.")
    
    # --- Sidebar for Control ---
    st.sidebar.header("Model Control")
    
    # Check if model exists
    model_exists = os.path.exists(MODEL_SAVE_PATH) and os.path.exists(CLASS_NAMES_PATH)
    
    if st.sidebar.button("Train Model", disabled=model_exists):
        with st.spinner("Training model, please wait... (This may take a few minutes)"):
            model, history = load_and_train_model()
            st.session_state['history'] = history
            st.session_state['model_trained'] = True
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model Status:** {'✅ Ready' if model_exists else '⏳ Needs Training'}")
    
    
    # --- Main Content Tabs ---
    tab_train, tab_predict = st.tabs(["Training & Evaluation", "Prediction"])
    
    with tab_train:
        st.header("Training Results")
        
        if 'history' in st.session_state or model_exists:
            if model_exists and 'history' not in st.session_state:
                st.warning("Model is saved, but training history is not available in this session. Re-run 'Train Model' to see the plots.")

            elif 'history' in st.session_state:
                plot_history(st.session_state['history'])
                
                # Display final metrics
                final_acc = st.session_state['history'].history['val_accuracy'][-1] * 100
                final_loss = st.session_state['history'].history['val_loss'][-1]
                st.metric(label="Validation Accuracy (Final)", value=f"{final_acc:.2f}%", delta=None)
                st.text(f"Validation Loss: {final_loss:.4f}")
        else:
            st.info("Click the 'Train Model' button in the sidebar to start training.")


    with tab_predict:
        st.header("Upload Image for Prediction")
        
        # Get class names for displaying dataframes
        current_class_names = get_class_names()
        
        if not model_exists and 'model_trained' not in st.session_state:
            st.warning("Please train the model first on the 'Training & Evaluation' tab.")
        elif not current_class_names:
            st.error("Cannot proceed to prediction: Class names could not be loaded from JSON. Please click 'Train Model' again.")
        else:
            # We explicitly check for both files here, then load them
            model = load_trained_model()
            
            if model is None:
                 st.warning("Model could not be loaded. Check console for errors.")
                 return # Exit early if model loading failed

            # REMOVED: st.image(Image.new('RGB', (1,1)), use_container_width=True) 

            uploaded_file = st.file_uploader(
                "Choose a corn image (.jpg, .jpeg, .png)", 
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                try:
                    # Read the image file
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(image, caption='Uploaded Image', use_container_width=True)

                    with col2:
                        st.subheader("Prediction Result")
                        # Make prediction
                        predicted_class, confidence, all_scores = predict_image(model, image)
                        
                        st.markdown(f"**Predicted Variety:** <span style='font-size: 24px; color: #10B981;'>{predicted_class}</span>", unsafe_allow_html=True)
                        
                        # REMOVED: st.markdown(f"**Confidence:** <span style='font-size: 24px; color: #10B981;'>{confidence*100:.2f}%</span>", unsafe_allow_html=True)
                        
                        # Show all class probabilities
                        st.markdown("---")
                        st.markdown("**All Class Probabilities:**")
                        
                        # Create a DataFrame for nice display
                        scores_df = tf.convert_to_tensor(all_scores).numpy()
                        scores_df = np.expand_dims(scores_df, axis=0)
                        df = (
                            tf.convert_to_tensor(scores_df) * 100
                        ).numpy().round(2)

                        st.dataframe(
                            data=df,
                            column_config={i: st.column_config.Progress(
                                label=current_class_names[i],
                                format="%f %%",
                                min_value=0.0,
                                max_value=100.0,
                            ) for i in range(len(current_class_names))},
                            hide_index=True,
                            use_container_width=True
                        )

                except Exception as e:
                    # This catches the specific "Class names are empty" error and others
                    st.error(f"Error processing image: {e}")


if __name__ == "__main__":
    main_app()