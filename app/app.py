import streamlit as st

st.set_page_config(page_title="Deepfake Detection", layout="centered")

st.title("🎭 Multimodal Deepfake Detection")
st.write("Demo application for detecting deepfake videos using audio and video analysis.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)
    st.success("Prediction: FAKE ❌ (demo output)")
