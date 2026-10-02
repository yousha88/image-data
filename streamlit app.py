import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PawVision AI",
    page_icon="🐾",
    layout="wide"
)


# =========================================================
# 2. CUSTOM CSS DESIGN
# =========================================================
st.markdown("""
<style>

/* Main page width */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Main heading */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 5px;
}


/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 19px;
    color: #888888;
    margin-bottom: 12px;
}


/* Welcome text */
.welcome-box {
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 30px;
    border: 1px solid rgba(128,128,128,0.25);
}


/* Prediction result */
.result-box {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.30);
    margin-top: 15px;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}

.confidence-text {
    font-size: 18px;
    margin-top: 8px;
}


/* Footer */
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================
MODEL_PATH = Path(__file__).parent / "cat_dog_model.keras"


@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_ai_model()

except Exception as e:
    st.error("❌ AI model could not be loaded.")
    st.write("Please make sure `cat_dog_model.keras` is in the same folder.")
    st.code(str(e))
    st.stop()


# =========================================================
# 4. HEADER / WELCOME
# =========================================================
st.markdown(
    '<div class="main-title">🐾 PawVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Cat vs Dog Image Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="welcome-box">

### 👋 Welcome!

Upload a **Cat 🐱 or Dog 🐶 image** and our AI model will analyze it.

You will also get useful information about your uploaded image,
including its **resolution, format, color mode, file size and aspect ratio**.

</div>
""", unsafe_allow_html=True)


# =========================================================
# 5. UPLOAD SECTION
# =========================================================
st.subheader("📤 Upload Your Image")

uploaded_file = st.file_uploader(
    "Choose a Cat or Dog image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# =========================================================
# 6. IF USER UPLOADS IMAGE
# =========================================================
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error("❌ This file could not be opened as an image.")
        st.stop()


    # -----------------------------------------------------
    # IMAGE INFORMATION
    # -----------------------------------------------------

    original_width, original_height = image.size

    file_size_kb = uploaded_file.size / 1024

    file_extension = uploaded_file.name.split(".")[-1].upper()

    aspect_ratio = original_width / original_height


    st.divider()

    st.subheader("🖼️ Image Preview & Details")


    # Two-column layout
    col1, col2 = st.columns([1.4, 1])


    # LEFT SIDE = IMAGE
    with col1:

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )


    # RIGHT SIDE = DETAILS
    with col2:

        st.markdown("### 📊 Image Information")

        st.metric(
            label="Resolution",
            value=f"{original_width} × {original_height}"
        )

        st.metric(
            label="File Size",
            value=f"{file_size_kb:.2f} KB"
        )

        st.metric(
            label="Format",
            value=file_extension
        )

        st.metric(
            label="Color Mode",
            value="RGB"
        )

        st.metric(
            label="Aspect Ratio",
            value=f"{aspect_ratio:.2f}"
        )


    # -----------------------------------------------------
    # EXTRA INFORMATION
    # -----------------------------------------------------

    with st.expander("🔎 View Technical Image Details"):

        st.write(f"**File Name:** {uploaded_file.name}")

        st.write(
            f"**Original Width:** {original_width} pixels"
        )

        st.write(
            f"**Original Height:** {original_height} pixels"
        )

        st.write(
            f"**Original Resolution:** "
            f"{original_width * original_height:,} pixels"
        )

        st.write(
            f"**Model Input Size:** 224 × 224 pixels"
        )

        st.write(
            "**Color Channels:** 3 (Red, Green, Blue)"
        )


    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    st.divider()

    st.subheader("🤖 AI Prediction")

    st.write(
        "Click the button below to let the trained AI model "
        "analyze your image."
    )


    if st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 AI is analyzing the image..."
        ):

            # Resize image exactly like training input
            resized_image = image.resize(
                (224, 224)
            )

            # Convert image to numbers
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            # Add batch dimension
            image_batch = np.expand_dims(
                image_array,
                axis=0
            )

            # Model prediction
            prediction = model.predict(
                image_batch,
                verbose=0
            )

            score = float(
                prediction[0][0]
            )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if score >= 0.5:

            predicted_class = "DOG 🐶"

            confidence = score * 100

            message = (
                "The AI model believes this image "
                "contains a dog."
            )

        else:

            predicted_class = "CAT 🐱"

            confidence = (1 - score) * 100

            message = (
                "The AI model believes this image "
                "contains a cat."
            )


        st.success(
            "✅ Analysis completed successfully!"
        )


        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-title">
                    {predicted_class}
                </div>

                <div class="confidence-text">
                    Model Confidence:
                    <b>{confidence:.2f}%</b>
                </div>

                <br>

                {message}

            </div>
            """,
            unsafe_allow_html=True
        )


        # Confidence progress bar
        st.write("### 🎯 Confidence Level")

        st.progress(
            min(
                max(confidence / 100, 0.0),
                1.0
            )
        )


        # -------------------------------------------------
        # MODEL SCORE DETAILS
        # -------------------------------------------------

        with st.expander(
            "🧪 View Prediction Details"
        ):

            st.write(
                f"**Raw Model Score:** {score:.6f}"
            )

            st.write(
                "**Decision Threshold:** 0.50"
            )

            st.write(
                "**Class Mapping:**"
            )

            st.write(
                "`0 = Cat`"
            )

            st.write(
                "`1 = Dog`"
            )

            st.info(
                "A score below 0.50 is classified as Cat, "
                "while a score of 0.50 or above is classified as Dog."
            )


else:

    st.info(
        "👆 Please upload a JPG, JPEG or PNG image to begin."
    )


# =========================================================
# 7. ABOUT THE MODEL
# =========================================================

st.divider()

st.subheader("🧠 About the AI Model")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "MobileNetV2"
    )

with col2:

    st.metric(
        "Task",
        "Binary Classification"
    )

with col3:

    st.metric(
        "Input",
        "224 × 224 RGB"
    )


# =========================================================
# 8. HOW APP WORKS
# =========================================================

with st.expander(
    "ℹ️ How does PawVision AI work?"
):

    st.markdown("""
### Step 1 — Upload
The user uploads a Cat or Dog image.

### Step 2 — Image Processing
The image is converted to **RGB** and resized to **224 × 224 pixels**.

### Step 3 — AI Analysis
The trained **MobileNetV2 Transfer Learning model** analyzes image features.

### Step 4 — Prediction
The model returns a score between **0 and 1**.

- Score `< 0.5` → **Cat 🐱**
- Score `≥ 0.5` → **Dog 🐶**

### Step 5 — Result
The app displays the predicted class and model confidence.
""")


# =========================================================
# 9. IMPORTANT NOTE
# =========================================================

with st.expander(
    "⚠️ Model Limitations"
):

    st.warning("""
This AI model was trained specifically to distinguish
between **Cats and Dogs**.

If you upload a person, car, bird, building or another object,
the model will still try to classify it as either Cat or Dog.

The displayed confidence is the model's output score and should
not be treated as a guaranteed real-world probability.
""")


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🐾 PawVision AI • Cat vs Dog Image Classification  
Built with TensorFlow, MobileNetV2 and Streamlit

</div>
""", unsafe_allow_html=True)