import os
import re
import json
import pytesseract
import uvicorn
import spacy
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from transformers import (pipeline,
                         LayoutLMv3ForTokenClassification,
                         LayoutLMv3Processor,
                         PegasusForConditionalGeneration,
                         PegasusTokenizer,
                         T5ForConditionalGeneration,
                         T5Tokenizer)
import torch
from concurrent.futures import ThreadPoolExecutor

# Initialize FastAPI app
app = FastAPI(title="Document Understanding and Summarization Engine")

# Configuration
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OUTPUT_DIR = "processed_documents"
    MODEL_CACHE_DIR = "model_cache"

    print(f"Tesseract path exists: {os.path.exists(TESSERACT_PATH)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model cache directory: {MODEL_CACHE_DIR}")
    
    LAYOUTLM_MODEL = "microsoft/layoutlmv3-base"
    PEGASUS_MODEL = "google/pegasus-xsum"
    T5_MODEL = "t5-small"
    try:
        SPACY_MODEL = spacy.load('en_core_web_lg')
        print(f"SPACY MODEL LOADED: {SPACY_MODEL}")
    except OSError:
        raise RuntimeError("SpaCy model 'en_core_web_lg' is not installed. Run: python -m spacy download en_core_web_lg")
    print(f"LAYOUTLM MODEL path exists: {os.path.exists(LAYOUTLM_MODEL)}")
    print(f"PEGASUS MODEL: {PEGASUS_MODEL}")
    print(f"T5 MODEL: {T5_MODEL}")
    print(f"SPACY MODEL: {SPACY_MODEL}")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH

class Models:
    _instance = None
    
    def __init__(self):
        self.spacy_model = Config.SPACY_MODEL
        self.nlp = None
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
            )
            self.layoutlm_model.to(Config.DEVICE)
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
            )
            self.pegasus_model.to(Config.DEVICE)
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
            )
            self.t5_model.to(Config.DEVICE)
        return self.t5_model, self.t5_tokenizer

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
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = pytesseract.image_to_string(Image.open(file_path))
            elif file_path.lower().endswith('.pdf'):
                text = pytesseract.image_to_string(Image.open(file_path))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"OCR processing failed: {str(e)}")
    
    def extract_entities_with_layoutlm(self, image_path: str) -> Dict:
        try:
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LayoutLM processing failed: {str(e)}")
    
    def extract_entities_with_spacy(self, text: str) -> Dict:
        try:
            nlp = self.models.load_spacy()
            doc = nlp(text)
            entities = {}
            for ent in doc.ents:
                entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
            return entities
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SpaCy NER processing failed: {str(e)}")
    
    def extract_custom_fields(self, text: str) -> Dict:
        results = {}
        for field_name, pattern in self.custom_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results[field_name] = matches[0] if len(matches) == 1 else matches
        return results
    
    def generate_summary_pegasus(self, text: str, max_length: int = 150) -> str:
        try:
            model, tokenizer = self.models.load_pegasus()
            inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(Config.DEVICE)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pegasus summarization failed: {str(e)}")
    
    def generate_summary_t5(self, text: str, max_length: int = 150) -> str:
        try:
            model, tokenizer = self.models.load_t5()
            input_text = "summarize: " + text
            inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True).to(Config.DEVICE)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"T5 summarization failed: {str(e)}")

# Utility function to ensure custom_fields values are strings
def sanitize_field(value: Union[str, List[str], None]) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        for v in value:
            if v:
                return str(v)
        return ""
    return str(value)

class DocumentResponse(BaseModel):
    document_type: str
    extracted_text: str
    entities: Dict[str, List[str]]
    custom_fields: Dict[str, str]
    summary: Optional[str]
    structured_data: Dict
    output_formats: List[str]

@app.post("/process-document/", response_model=DocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    document_type: str = "invoice",
    generate_summary: bool = True,
    output_format: str = "json"
):
    try:
        processor = DocumentProcessor()
        file_path = os.path.join(Config.OUTPUT_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        with ThreadPoolExecutor() as executor:
            text_future = executor.submit(processor.extract_text_with_ocr, file_path)
            
            layoutlm_future = None
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                layoutlm_future = executor.submit(processor.extract_entities_with_layoutlm, file_path)
            
            text = text_future.result()
            spacy_future = executor.submit(processor.extract_entities_with_spacy, text)
            custom_fields_future = executor.submit(processor.extract_custom_fields, text)
            
            summary_future = None
            if generate_summary:
                summary_future = executor.submit(processor.generate_summary_pegasus, text)
            
            entities = spacy_future.result()
            custom_fields = custom_fields_future.result()
            
            if layoutlm_future:
                layoutlm_entities = layoutlm_future.result()
                for k, v in layoutlm_entities.items():
                    entities[k] = entities.get(k, []) + v
            
            summary = summary_future.result() if summary_future else None
        
        structured_data = {
            "metadata": {
                "filename": file.filename,
                "document_type": document_type
            },
            "entities": entities,
            "custom_fields": custom_fields,
            "summary": summary
        }
        
        output_files = []
        base_name = os.path.splitext(file.filename)[0]
        
        if output_format in ["json", "all"]:
            json_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.json")
            with open(json_path, "w") as f:
                json.dump(structured_data, f, indent=2)
            output_files.append(json_path)
        
        if output_format in ["excel", "all"]:
            excel_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.xlsx")
            df_data = []
            for entity_type, values in entities.items():
                for value in values:
                    df_data.append({"Entity Type": entity_type, "Value": value})
            for field, value in custom_fields.items():
                if isinstance(value, list):
                    for v in value:
                        df_data.append({"Entity Type": field, "Value": v})
                else:
                    df_data.append({"Entity Type": field, "Value": value})
            df = pd.DataFrame(df_data)
            with pd.ExcelWriter(excel_path) as writer:
                df.to_excel(writer, sheet_name="Entities", index=False)
                if summary:
                    pd.DataFrame({"Summary": [summary]}).to_excel(writer, sheet_name="Summary", index=False)
            output_files.append(excel_path)
        
        if output_format in ["pdf", "all"]:
            pdf_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.pdf")
            output_files.append(pdf_path)
        
        # Clean custom_fields to strings
        cleaned_custom_fields = {k: sanitize_field(v) for k, v in custom_fields.items()}
        
        response_data = {
            "document_type": document_type,
            "extracted_text": text,
            "entities": entities,
            "custom_fields": cleaned_custom_fields,
            "summary": summary,
            "structured_data": structured_data,
            "output_formats": output_files
        }
        
        return DocumentResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

@app.get("/download-result/")
async def download_result(filename: str):
    file_path = os.path.join(Config.OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
