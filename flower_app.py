import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --- Load model and class indices only once ---
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("flower_species_cnn.h5")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # invert {class: index} -> {index: class}
    inv_class_indices = {int(v): k for k, v in class_indices.items()}
    return model, inv_class_indices

model, class_labels = load_model_and_classes()

st.title("🌸 Flower Species Classifier")
st.write("Upload a flower image and I’ll predict its species!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img_arr = image.resize((224, 224))
    img_arr = np.expand_dims(np.array(img_arr) / 255.0, axis=0)

    # Predict
    pred = model.predict(img_arr)
    predicted_index = int(np.argmax(pred))
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(pred))

    st.success(f"🌼 Predicted: **{predicted_label}** ({confidence:.1%} confidence)")
