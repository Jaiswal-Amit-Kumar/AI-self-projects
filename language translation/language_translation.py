from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
import os
import shutil
import torch
import uvicorn
from tempfile import NamedTemporaryFile
import sounddevice as sd
from scipy.io.wavfile import write
import uuid
from datetime import datetime
import logging
import traceback
from typing import List
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("translation-api")

app = FastAPI()

# Device and precision
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"Running on device: {device}, compute type: {compute_type}")

# Load Whisper model for speech-to-text
whisper_model = WhisperModel(
    "large-v3",
    device=device,
    compute_type=compute_type,
    download_root="./whisper_cache"
)

# Indic languages and their ISO-639-1 codes for normalization
lang_code_map = {
    "hindi": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "kannada": "kn",
    "assamese": "as",
    "odia": "or",
    "marathi": "mr",
    "punjabi": "pa",
    "gujarati": "gu",
    "sanskrit": "sa"
}

# IndicTrans2 language codes required for translation model input tags
indictrans_code_map = {
    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
    "te": "tel",
    "ml": "mal",
    "kn": "kan",
    "as": "asm",
    "or": "ori",
    "mr": "mar",
    "pa": "pan",
    "gu": "guj",
    "sa": "san"
}

# Load translation model and tokenizer from AI4Bharat IndicTrans2
model_name = "ai4bharat/indictrans2-indic-en-1B"
logger.info("Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
logger.info("Translation model loaded successfully")

tokenizer.src_lang = "hi"
tokenizer.tgt_lang = "en"



# Warm up the model - FIXED FORMAT
logger.info("Warming up translation model...")
with torch.no_grad():
    # Correct format: <src_lang> <tgt_lang> <text>
    warmup_text = tokenizer(["hin ||| eng ||| नमस्ते"], return_tensors="pt", padding=True).to(device)
    model.generate(**warmup_text, max_length=128)
logger.info("Model warmup complete")

# Setup IndicNLP normalizers for each supported Indic language
factory = IndicNormalizerFactory()
normalizers = {lang: factory.get_normalizer(code) for lang, code in lang_code_map.items()}


class TranslationRequest(BaseModel):
    text: str = Field(..., title="Text to Translate", example="नमस्ते, आप कैसे हैं?")
    source_language: str = Field(..., title="Source Language", example="Hindi")

    @validator('source_language')
    def validate_source_language(cls, v):
        if v not in lang_code_map:
            raise ValueError(f"Unsupported source language '{v}'. Supported languages: {list(lang_code_map.keys())}")
        return v


@app.get("/")
def root():
    return {"message": "🎙️ Voice Language Recognition and Translation API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device}


@app.post("/record-and-transcribe/", response_class=HTMLResponse)
async def record_and_transcribe(duration: int = Query(5, ge=1, le=60, description="Recording duration in seconds (1-60)")):
    try:
        fs = 16000  # Sampling frequency
        logger.info(f"Recording audio for {duration} seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        tmp_path = f"mic_recording_{uuid.uuid4().hex}.wav"
        write(tmp_path, fs, recording)

        logger.info("Transcribing audio...")
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        transcription = " ".join(seg.text for seg in segments)
        detected_lang = info.language

        os.remove(tmp_path)

        html = f"""
        <html>
        <head><title>Transcription Report</title></head>
        <body style="font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px;">
            <h2>📝 Transcription Report</h2>
            <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><b>Duration:</b> {duration} seconds</p>
            <p><b>Detected Language:</b> {detected_lang}</p>
            <p><b>Transcription:</b><br><pre style="background:#eee;padding:10px;border-radius:5px;">{transcription}</pre></p>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        return HTMLResponse(content=f"<h3 style='color:red'>Error: {str(e)}</h3>", status_code=500)


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["audio/wav", "audio/mpeg", "audio/x-wav"]:
            raise HTTPException(400, "Unsupported file format. Supported: wav, mp3")

        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"Transcribing audio file: {file.filename}")
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        transcription = " ".join(seg.text for seg in segments)
        detected_lang = info.language

        os.remove(tmp_path)

        return {
            "filename": file.filename,
            "detected_language": detected_lang,
            "transcription": transcription,
            "duration": info.duration
        }

    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(500, f"Error processing audio: {str(e)}")


def preprocess_text(text: str, iso_code: str) -> str:
    """Normalize and tokenize Indic text using ISO-639-1 code"""
    normalizer = normalizers.get(iso_code)
    if normalizer:
        normalized = normalizer.normalize(text)
        tokenized = " ".join(indic_tokenize.trivial_tokenize(normalized, iso_code))
        return tokenized
    return text


# Dummy placeholders for demonstration
tokenizer = None
model = None
device = torch.device("cpu")

# Helper function to preprocess text (dummy, replace with your logic)
def preprocess_text(text: str, iso_code: str) -> str:
    return text.strip()

def clean_translation(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"\s([,.!?])", r"\1", text)
    return text

# --- Pydantic model for batch translation ---
class BatchTranslationRequest(BaseModel):
    texts: List[str] = Field(..., title="List of texts to translate")
    source_language: str = Field(..., title="Source Language", example="Hindi")

    @validator('source_language')
    def validate_source_language(cls, v):
        if v not in lang_code_map:
            raise ValueError(f"Unsupported source language '{v}'. Supported languages: {list(lang_code_map.keys())}")
        return v

@app.post("/batch-translate-to-english/")
async def batch_translate(request: BatchTranslationRequest):
    iso_code = lang_code_map[request.source_language]
    source_tag = f"<{indictrans_code_map[iso_code]}>"
    target_tag = "<en>"

    translations = []
    for text in request.texts:
        processed_text = preprocess_text(text, iso_code)
        input_text = f"{source_tag} {target_tag} {processed_text}"

        # Tokenize and translate - Replace below with your model inference
        # Example:
        # inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        # with torch.no_grad():
        #     output = model.generate(**inputs, max_length=512)
        # translated = tokenizer.decode(output[0], skip_special_tokens=True)

        tokenizer.src_lang = indictrans_code_map[iso_code]  # E.g., "hin"
        tokenizer.tgt_lang = "en"

        inputs = tokenizer([processed_text], return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=512)

        translated = tokenizer.batch_decode(output, skip_special_tokens=True)[0]


        translated = clean_translation(translated)
        translations.append({"original": text, "translated": translated})

    return JSONResponse({
        "source_language": request.source_language,
        "translations": translations
    })


@app.get("/translate-form", response_class=HTMLResponse)
def translate_form():
    options = "\n".join(f'<option value="{lang}">{lang}</option>' for lang in lang_code_map.keys())
    html = f"""
    <html>
        <head>
            <title>Translate to English</title>
        </head>
        <body>
            <h2>Translate Text to English</h2>
            <form method="post" action="/translate-to-english/">
                <label for="text">Text:</label><br>
                <textarea id="text" name="text" rows="4" cols="50" required></textarea><br><br>

                <label for="source_language">Source Language:</label><br>
                <select id="source_language" name="source_language" required>
                    {options}
                </select><br><br>

                <input type="submit" value="Translate">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/translate-to-english/")
async def translate_to_english(
    text: str = Form(...),
    source_language: str = Form(...)
):
    if source_language not in lang_code_map:
        raise HTTPException(400, f"Unsupported source language '{source_language}'")

    iso_code = lang_code_map[source_language]
    processed_text = preprocess_text(text, iso_code)
    source_tag = f"<{indictrans_code_map[iso_code]}>"
    target_tag = "<en>"
    input_text = f"{source_tag} {target_tag} {processed_text}"

    # Tokenize and translate - Replace below with your model inference
    # Example:
    # inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    # with torch.no_grad():
    #     output = model.generate(**inputs, max_length=512)
    # translated = tokenizer.decode(output[0], skip_special_tokens=True)

    translated = f"Translated({text})"  # Dummy translation placeholder

    translated = clean_translation(translated)

    html = f"""
    <html>
        <head><title>Translation Result</title></head>
        <body>
            <h2>Translation Result</h2>
            <p><b>Original Text:</b> {text}</p>
            <p><b>Source Language:</b> {source_language}</p>
            <p><b>Translated Text:</b> {translated}</p>
            <br>
            <a href="/translate-form">Translate Another</a>
        </body>
    </html>
    """
    return HTMLResponse(content=html)



@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = b""
    chunk_size = 16000 * 2  # 1 second buffer for 16-bit mono PCM

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data

            if len(buffer) >= chunk_size:
                with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(buffer[:chunk_size])
                    tmp_path = tmp.name

                buffer = buffer[chunk_size:]

                segments, _ = whisper_model.transcribe(tmp_path, beam_size=5, vad_filter=True)
                transcription = " ".join(seg.text for seg in segments)
                os.remove(tmp_path)

                await websocket.send_json({
                    "transcription": transcription,
                    "timestamp": datetime.now().isoformat()
                })

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"error": str(e)})

    finally:
        await websocket.close()


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down, releasing resources...")
    # Clean up models to free memory
    global model, whisper_model
    del model
    del whisper_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Resources released")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=300,
        workers=1,
        log_level="info"
    )