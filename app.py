
import streamlit as st
import keras
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")

# Load the trained model
@st.cache_resource
def load_model():
    return keras.models.load_model("pneumonia_model.keras")

model = load_model()

# Labels: 0 = NORMAL, 1 = PNEUMONIA
class_names = ["NORMAL", "PNEUMONIA"]

st.title("🫁 Chest X-Ray Classification")
st.write("Upload a chest X-ray image to preprocess it and get a model prediction.")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open and convert the uploaded image to RGB
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Original Image", width=350)

    # Save temporarily because load_img accepts a file path
    with tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=False
    ) as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    try:
        # Resize image to the model's expected input size
        resized_image = keras.utils.load_img(
            image_path,
            # target_size=(180, 180),
            color_mode="rgb"
        )

        # Convert image to a NumPy array
        image_array = keras.utils.img_to_array(resized_image)

        st.success("Image processed successfully!")

        image_batch = np.expand_dims(image_array, axis=0)

        # Predict
        if st.button("Classify X-Ray"):
            with st.spinner("Analyzing image..."):
                prediction = model.predict(image_batch, verbose=0)

                # Assumes sigmoid output: probability of class 1
                probability = float(prediction[0][0])

                predicted_class = (
                    1 if probability >= 0.5 else 0
                )

                confidence = (
                    probability if predicted_class == 1
                    else 1 - probability
                )

            st.subheader("Prediction")

            if predicted_class == 1:
                st.error("Prediction: PNEUMONIA")
            else:
                st.success("Prediction: NORMAL")

            st.write(f"Confidence: {confidence:.2%}")

            st.write(
                f"Pneumonia probability: {probability:.2%}"
            )

            st.progress(probability)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)
