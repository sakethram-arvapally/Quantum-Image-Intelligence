import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageEnhance
import time

st.set_page_config(page_title="Quantum Neural Network for Image Intelligence",
                   page_icon="🧠", layout="wide")

st.title("Quantum Neural Network for Large-Scale Image Intelligence")
st.caption("Academic prototype: classical image classification + simulated quantum feature circuit")

st.info(
    "Honest scope: the classical result uses a pretrained MobileNetV2 model. "
    "The 4-qubit circuit below is a simulator demonstration with untrained parameters; "
    "its measurements are quantum features, not a trained QNN prediction."
)

@st.cache_resource(show_spinner="Loading the pretrained image model...")
def load_classical_model():
    import tensorflow as tf
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    resized = image.convert("RGB").resize((224, 224))
    arr = np.asarray(resized, dtype=np.float32)
    # MobileNetV2 preprocessing: scale RGB pixels from [0, 255] to [-1, 1].
    model_input = (arr.copy() / 127.5) - 1.0
    return resized, arr, np.expand_dims(model_input, axis=0)

def image_features(arr):
    # Values are calculated from the actual uploaded image.
    rgb_means = arr.mean(axis=(0, 1)) / 255.0
    brightness = float(arr.mean() / 255.0)
    contrast = float(arr.std() / 255.0)
    # Four bounded values used as rotation angles in the circuit.
    features = np.array([
        rgb_means[0] * np.pi,
        rgb_means[1] * np.pi,
        rgb_means[2] * np.pi,
        min(1.0, contrast * 3.0) * np.pi
    ], dtype=float)
    return rgb_means, brightness, contrast, features

def run_quantum_demo(features):
    import pennylane as qml
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    # Fixed seeded parameters make the demonstration repeatable.
    rng = np.random.default_rng(7)
    weights = rng.uniform(-np.pi, np.pi, size=(2, n_qubits, 3))

    @qml.qnode(dev)
    def circuit(x):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
            qml.RZ(x[i] / 2.0, wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for layer in range(weights.shape[0]):
            for i in range(n_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    measurements = np.asarray(circuit(features), dtype=float)
    return measurements, qml.draw(circuit)(features)

with st.sidebar:
    st.header("Project workflow")
    st.markdown("1. Upload image\n2. Preprocess to 224 × 224\n3. Classify with MobileNetV2\n4. Encode image features into 4 qubits\n5. Simulate gates and measurements")
    st.caption("First run requires internet to download pretrained MobileNetV2 weights.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.subheader("How to test")
    st.write("Upload a clear image of an everyday object, animal, plant, food item, or vehicle.")
    st.write("The pretrained classifier supports ImageNet categories; it may not recognize every image correctly.")
else:
    try:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Input image")
            st.image(image, use_container_width=True)
            st.write(f"Original dimensions: **{image.width} × {image.height}**")
        with col2:
            st.subheader("Preprocessing preview")
            resized, arr, model_input = preprocess_image(image)
            st.image(resized, caption="Resized to 224 × 224 and converted to RGB", use_container_width=True)

        rgb_means, brightness, contrast, q_features = image_features(arr)
        st.subheader("Image features")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Mean red", f"{rgb_means[0]:.3f}")
        f2.metric("Mean green", f"{rgb_means[1]:.3f}")
        f3.metric("Mean blue", f"{rgb_means[2]:.3f}")
        f4.metric("Contrast", f"{contrast:.3f}")

        if st.button("Analyze Image", type="primary", use_container_width=True):
            try:
                import tensorflow as tf
                start = time.perf_counter()
                model = load_classical_model()
                predictions = model.predict(model_input, verbose=0)
                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
                classical_seconds = time.perf_counter() - start

                with st.spinner("Simulating the 4-qubit feature circuit..."):
                    q_start = time.perf_counter()
                    measurements, circuit_text = run_quantum_demo(q_features)
                    quantum_seconds = time.perf_counter() - q_start

                st.success("Analysis completed.")
                left, right = st.columns(2)
                with left:
                    st.subheader("Classical baseline: MobileNetV2")
                    result_df = pd.DataFrame(
                        [{"Class": label.replace("_", " "), "Score (%)": round(float(score) * 100, 2)}
                         for _, label, score in decoded]
                    )
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.metric("Inference time", f"{classical_seconds:.3f} s")
                    st.caption("Scores are pretrained model probabilities, not project-trained results.")
                with right:
                    st.subheader("Simulated quantum feature circuit")
                    q_df = pd.DataFrame({
                        "Qubit": [f"Q{i}" for i in range(4)],
                        "Input rotation (radians)": np.round(q_features, 4),
                        "⟨Z⟩ measurement": np.round(measurements, 4)
                    })
                    st.dataframe(q_df, use_container_width=True, hide_index=True)
                    st.metric("Circuit simulation time", f"{quantum_seconds:.3f} s")
                    st.caption("Measurements depend on image features and fixed, untrained circuit parameters.")

                with st.expander("View quantum circuit structure"):
                    st.code(circuit_text, language="text")
                with st.expander("Testing and validation notes"):
                    st.markdown(
                        "- Change the uploaded image and verify that image features and circuit measurements change.\n"
                        "- Check that the classical model returns five ranked ImageNet labels.\n"
                        "- Record predictions and timings for several images in your review notes.\n"
                        "- These tests demonstrate application functionality, not scientific proof of quantum advantage.\n"
                        "- To claim a trained QNN classifier, train circuit parameters on a labeled dataset and evaluate against a classical baseline."
                    )
            except ImportError as e:
                st.error("A required package is missing. Install dependencies from requirements.txt and restart the app.")
                st.code(str(e))
            except Exception as e:
                st.error("Analysis failed. Check the installation and internet connection, then try again.")
                st.code(str(e))
    except Exception as e:
        st.error(f"Could not open this image: {e}")

st.divider()
st.subheader("Project pipeline")
st.markdown("**Image input → preprocessing → classical classification → image-feature encoding → 4-qubit simulation → measurement visualization**")
st.caption("This prototype demonstrates a hybrid workflow. It does not yet train or validate a quantum neural network for large-scale classification.")
