import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image
import datetime
import io
from gtts import gTTS
import os
import threading
import queue

# Initialize session state for voice settings
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = True

# Queue for thread-safe audio playback
audio_queue = queue.Queue()

# Load the pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

# Preprocess the uploaded image for model input
def preprocess_image(image):
    img = np.array(image.convert("RGB"))  # Convert to RGB
    img = cv2.resize(img, (224, 224))     # Resize
    img = preprocess_input(img)           # Preprocess for MobileNetV2
    img = np.expand_dims(img, axis=0)     # Add batch dimension
    return img

# Classify the image using the loaded model
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded = decode_predictions(predictions, top=3)[0]
        return decoded
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

# Generate speech audio file (thread-safe)
def generate_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = f"temp_{threading.get_ident()}.mp3"
        tts.save(temp_file)
        audio_queue.put(temp_file)
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")

# Speak out the top prediction in a separate thread
def speak_prediction(label):
    if not st.session_state.voice_enabled:
        return
        
    text = f"This looks like a {label.replace('_', ' ')}"
    threading.Thread(target=generate_speech, args=(text,)).start()

# Save prediction report
def save_report(predictions):
    report = io.StringIO()
    report.write("AI Image Classifier - Prediction Report\n")
    report.write("-" * 40 + "\n")
    report.write(f"Generated at: {datetime.datetime.now()}\n\n")
    report.write("Top Predictions:\n")
    for i, (_, label, score) in enumerate(predictions):
        report.write(f"{i+1}. {label.replace('_', ' ')}: {score:.2%}\n")
    report_content = report.getvalue()
    report.close()
    return report_content

# Main Streamlit app
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

    st.title("🖼️ AI Image Classifier")
    st.markdown("Upload an image and let AI tell you what it sees!\n\nBuilt with MobileNetV2 and Streamlit.")

    # Voice toggle in sidebar
    with st.sidebar:
        st.header("Settings")
        voice_toggle = st.toggle("Enable Voice Narration", 
                                value=st.session_state.voice_enabled,
                                help="Turn on/off voice descriptions of predictions")
        
        if voice_toggle != st.session_state.voice_enabled:
            st.session_state.voice_enabled = voice_toggle
            st.rerun()

    # Check for any generated audio files
    while not audio_queue.empty():
        audio_file = audio_queue.get()
        if os.path.exists(audio_file):
            with open(audio_file, "rb") as audio_binary:
                st.audio(audio_binary.read(), format="audio/mp3")
            os.remove(audio_file)

    # Cache model
    model = load_model()

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify Image"):
            with st.spinner("Analyzing Image..."):
                image_obj = Image.open(uploaded_file)
                predictions = classify_image(model, image_obj)

            if predictions:
                st.subheader("Predictions")
                for i, (_, label, score) in enumerate(predictions):
                    display_label = label.replace('_', ' ')
                    st.write(f"**{i+1}. {display_label}**: {score:.2%}")
                    st.progress(min(int(score * 100), 100))

                # Speak top prediction in background if enabled
                # Speak top prediction using browser's speech synthesis (better UX)
                top_label = predictions[0][1]
                if st.session_state.voice_enabled:
                    speak_text = f"This looks like a {top_label.replace('_', ' ')}"
                    st.markdown(f"""
                    <script>
                    var utterance = new SpeechSynthesisUtterance("{speak_text}");
                    window.speechSynthesis.speak(utterance);
                </script>
                 """, unsafe_allow_html=True)

                # Generate report content
                report_content = save_report(predictions)
                
                # Download button
                st.download_button(
                    label="Download Prediction Report",
                    data=report_content,
                    file_name=f"Prediction_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
