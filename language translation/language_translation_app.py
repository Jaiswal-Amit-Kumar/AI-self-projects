import streamlit as st
import torch
import os
import shutil
import uuid
import sounddevice as sd
from scipy.io.wavfile import write
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel
from transformers import pipeline, MarianTokenizer
import numpy

# ------------------ SETUP ------------------
st.set_page_config(page_title="Voice Transcriber", layout="centered")
st.title("🎤 Voice Recorder and Transcriber")

# Load whisper model once
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel("large-v3", device=device, compute_type=compute_type)

model = load_model()

# Load translation pipeline once
@st.cache_resource
def load_translation_pipeline(target_lang):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-ROMANCE{target_lang}")

# Supported output languages
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Arabic": "ar",
    "Chinese": "zh"
}

# Initialize session state to persist values across interactions
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "language" not in st.session_state:
    st.session_state.language = ""

# ------------------ UI ------------------
duration = st.slider("🎚️ Recording duration (seconds)", 1, 10, 5)

choice = st.selectbox(
    "Choose input method",
    ["Live Real time audio recording", "Pre-recorded audio"],
    index=None,
    placeholder="Select input method..."
)

if choice == "Live Real time audio recording":
    if st.button("🎙️ Start Recording"):
        fs = 16000
        st.info("Recording... Please speak clearly.")
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            tmp_file = f"record_{uuid.uuid4().hex}.wav"
            write(tmp_file, fs, recording)

            st.info("Transcribing...")
            segments, info = model.transcribe(tmp_file, beam_size=5, language=None)
            transcription = " ".join([seg.text for seg in segments])
            detected_language = info.language
            os.remove(tmp_file)

            # Save to session state
            st.session_state.transcription = transcription
            st.session_state.language = detected_language
            st.session_state.show_translation = True  # <-- Add this


            st.success(f"🌐 Detected Language: {detected_language}")
            st.markdown("### 📝 Transcription:")
            st.markdown(
                f"<div style='background:#000000;padding:15px;border-radius:8px;font-size:16px;color:white;'>{transcription}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")

elif choice == "Pre-recorded audio":
    uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
    
    if uploaded_file:
        st.write("📁 File name:", uploaded_file.name)

        if st.button("Transcribe"):
            try:
                with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    shutil.copyfileobj(uploaded_file, tmp)
                    tmp_path = tmp.name

                st.info("Transcribing...")
                segments, info = model.transcribe(tmp_path, beam_size=5, language=None)
                transcription = " ".join([seg.text for seg in segments])
                detected_language = info.language
                os.remove(tmp_path)
                
                # Save to session state
                st.session_state.transcription = transcription
                st.session_state.language = detected_language
                st.session_state.show_translation = True  # <-- Add this


                st.success(f"🌐 Detected Language: {detected_language}")
                st.markdown("### 📝 Transcription:")
                st.markdown(
                    f"<div style='background:#000000;padding:15px;border-radius:8px;font-size:16px;color:white;'>{transcription}</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"❌ Error with file {uploaded_file.name}: {str(e)}")

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ----------- Translation Section -------------
if st.session_state.get("show_translation", False):
    st.markdown("## 🌍 Translation")

    target_language = st.selectbox(
        "Choose your target language", 
        list(lang_map.keys()), 
        index=None, 
        placeholder="Select your target language"
    )

    
    if st.button("Translate"):
        transcription = st.session_state.get("transcription", "")
        
        if not transcription:
            st.warning("No transcription available to translate.")
        elif not target_language:
            st.warning("Please select a target language.")
        elif target_language == "English":
            st.subheader("🌍 Translated Text (English):")
            st.write(transcription)
        else:
            with st.spinner(f"Translating to {target_language}..."):
                try:
                    # Load translator pipeline
                    translator = load_translation_pipeline(lang_map[target_language])

                    # Split long transcriptions into manageable chunks
                    chunks = chunk_text(transcription, max_words=200)

                    # Use the pipeline directly
                    translations = [translator(chunk)[0]['translation_text'] for chunk in chunks]
                    translated_text = " ".join(translations)

                    st.subheader(f"🌍 Translated Text ({target_language}):")
                    st.write(translated_text)
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")


