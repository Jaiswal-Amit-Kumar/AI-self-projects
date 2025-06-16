import os
import re
import json
import pytesseract
import uvicorn
import spacy
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
import torch
from concurrent.futures import ThreadPoolExecutor
from fpdf import FPDF

# Initialize FastAPI app
app = FastAPI(title="Document Understanding and Summarization Engine")

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    OUTPUT_DIR = "processed_documents"
    MODEL_CACHE_DIR = "model_cache"

    LAYOUTLM_MODEL = "microsoft/layoutlmv3-base"
    PEGASUS_MODEL = "google/pegasus-xsum"
    T5_MODEL = "t5-small"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    try:
        SPACY_MODEL = spacy.load('en_core_web_lg')
    except OSError:
        raise RuntimeError("SpaCy model 'en_core_web_lg' is not installed. Run: python -m spacy download en_core_web_lg")

class Models:
    _instance = None

    def __init__(self):
        self.spacy_model = Config.SPACY_MODEL
        self.layoutlm_model = None
        self.layoutlm_processor = None
        self.pegasus_model = None
        self.pegasus_tokenizer = None
        self.t5_model = None
        self.t5_tokenizer = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_spacy(self):
        return self.spacy_model

    def load_layoutlm(self):
        if self.layoutlm_model is None:
            self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained(
                Config.LAYOUTLM_MODEL,
                cache_dir=Config.MODEL_CACHE_DIR
            ).to(Config.DEVICE)
            self.layoutlm_processor = LayoutLMv3Processor.from_pretrained(
                Config.LAYOUTLM_MODEL,
                cache_dir=Config.MODEL_CACHE_DIR
            )
        return self.layoutlm_model, self.layoutlm_processor

    def load_pegasus(self):
        if self.pegasus_model is None:
            self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(
                Config.PEGASUS_MODEL,
                cache_dir=Config.MODEL_CACHE_DIR
            )
            self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(
                Config.PEGASUS_MODEL,
                cache_dir=Config.MODEL_CACHE_DIR
            ).to(Config.DEVICE)
        return self.pegasus_model, self.pegasus_tokenizer

    def load_t5(self):
        if self.t5_model is None:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                Config.T5_MODEL,
                cache_dir=Config.MODEL_CACHE_DIR
            )
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                Config.T5_MODEL,
                cache_dir=Config.MODEL_CACHE_DIR
            ).to(Config.DEVICE)
        return self.t5_model, self.t5_tokenizer


def generate_pdf(file_path: str, summary: str, entities: Dict, custom_fields: Dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def safe_text(text):
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf.cell(0, 10, safe_text("Document Summary"), ln=True)
    pdf.multi_cell(0, 10, safe_text(summary or "No summary available."))
    pdf.ln(10)

    pdf.cell(0, 10, safe_text("Entities Extracted:"), ln=True)
    for entity_type, values in entities.items():
        pdf.cell(0, 10, safe_text(f"{entity_type}:"), ln=True)
        for v in values:
            pdf.cell(0, 10, safe_text(f" - {v}"), ln=True)
        pdf.ln(5)

    pdf.cell(0, 10, safe_text("Custom Fields:"), ln=True)
    for field, value in custom_fields.items():
        if isinstance(value, list):
            for v in value:
                pdf.cell(0, 10, safe_text(f" - {field}: {v}"), ln=True)
        else:
            pdf.cell(0, 10, safe_text(f"{field}: {value}"), ln=True)

    pdf.output(file_path)


class DocumentProcessor:
    def __init__(self):
        self.models = Models.get_instance()
        self.custom_patterns = {
            "invoice_number": r"(?:invoice|inv)\.?\s*#?\s*([A-Z0-9-]+)",
            "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            "amount": r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(\+?\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
        }

    def extract_text_with_ocr(self, file_path: str) -> str:
        try:
            return pytesseract.image_to_string(Image.open(file_path))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"OCR failed: {e}")

    def extract_entities_with_layoutlm(self, image_path: str) -> Dict:
        model, processor = self.models.load_layoutlm()
        image = Image.open(image_path).convert("RGB")
        encoding = processor(image, return_tensors="pt", truncation=True)
        encoding = {k: v.to(Config.DEVICE) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())

        results = {}
        current_entity = ""
        current_label = ""

        for token, prediction in zip(tokens, predictions):
            label = model.config.id2label[prediction]
            if token.startswith("##"):
                current_entity += token[2:]
            else:
                if current_entity:
                    results[current_label] = results.get(current_label, []) + [current_entity]
                current_entity = token
                current_label = label

        if current_entity:
            results[current_label] = results.get(current_label, []) + [current_entity]

        return results

    def extract_entities_with_spacy(self, text: str) -> Dict:
        doc = self.models.load_spacy()(text)
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
        return entities

    def extract_custom_fields(self, text: str) -> Dict:
        results = {}
        for field, pattern in self.custom_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results[field] = matches if len(matches) > 1 else [matches[0]]
        return results

    def generate_summary_pegasus(self, text: str, max_length=150) -> str:
        model, tokenizer = self.models.load_pegasus()
        inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(Config.DEVICE)
        summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class DocumentResponse(BaseModel):
    document_type: str
    extracted_text: str
    entities: Dict[str, List[str]]
    custom_fields: Dict[str, List[str]]
    summary: Optional[str]
    output_formats: List[str]


@app.post("/process-document/", response_model=DocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    document_type: str = Query(...),
    generate_summary: bool = Query(True),
    output_format: List[str] = Query(["json"])
):
    try:
        processor = DocumentProcessor()
        file_path = os.path.join(Config.OUTPUT_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        with ThreadPoolExecutor() as executor:
            text_future = executor.submit(processor.extract_text_with_ocr, file_path)
            layoutlm_future = executor.submit(processor.extract_entities_with_layoutlm, file_path)

            text = text_future.result()
            spacy_future = executor.submit(processor.extract_entities_with_spacy, text)
            custom_fields_future = executor.submit(processor.extract_custom_fields, text)
            summary_future = executor.submit(processor.generate_summary_pegasus, text) if generate_summary else None

            entities = spacy_future.result()
            custom_fields = custom_fields_future.result()

            layoutlm_entities = layoutlm_future.result()
            for k, v in layoutlm_entities.items():
                entities[k] = entities.get(k, []) + v

            summary = summary_future.result() if summary_future else None

        structured_data = {
            "metadata": {"filename": file.filename, "document_type": document_type},
            "entities": entities,
            "custom_fields": custom_fields,
            "summary": summary
        }

        base_name = os.path.splitext(file.filename)[0]
        output_files = []

        if "json" in output_format or "all" in output_format:
            json_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            output_files.append(f"{base_name}.json")

        if "excel" in output_format or "all" in output_format:
            excel_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.xlsx")
            df_data = [{"Entity Type": k, "Value": v} for k, vs in {**entities, **custom_fields}.items() for v in vs]
            pd.DataFrame(df_data).to_excel(excel_path, index=False)
            output_files.append(f"{base_name}.xlsx")

        if "pdf" in output_format or "all" in output_format:
            pdf_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.pdf")
            generate_pdf(pdf_path, summary or "", entities, custom_fields)
            output_files.append(f"{base_name}.pdf")

        return DocumentResponse(
            document_type=document_type,
            extracted_text=text,
            entities=entities,
            custom_fields=custom_fields,
            summary=summary,
            output_formats=output_files
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-result/")
async def download_result(filename: str):
    file_path = os.path.join(Config.OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/pdf')


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
