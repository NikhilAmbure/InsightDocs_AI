import logging
import fitz  # PyMuPDF (PDF)
import pytesseract  # OCR for images
import google.generativeai as genai

from docx import Document as DocxDocument  # DOCX
from pptx import Presentation  # PPTX
from PIL import Image  # Images

from langchain_text_splitters import RecursiveCharacterTextSplitter
from django.conf import settings
from ..models import DocumentChunk

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GOOGLE_API_KEY)

def extract_text_from_file(file_path, mime_type):
    """Extract text content based on file type."""
    text = ""
    try:
        if mime_type == 'application/pdf':
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        
        elif 'wordprocessing' in mime_type or mime_type == 'application/msword':
            doc = DocxDocument(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        elif mime_type == 'text/plain':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        # PowerPoint presentations (ppt / pptx)
        elif 'presentation' in mime_type or mime_type == 'application/vnd.ms-powerpoint':
            prs = Presentation(file_path)
            slide_texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_texts.append(shape.text)
            text = "\n\n".join(slide_texts)

        # Common image types -> OCR via Tesseract
        elif mime_type.startswith('image/'):
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
                
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return None
        
    return text

def process_document_for_rag(document, file_path):
    """Chunk document and save embeddings."""
    try:
        # 1. Extract Text
        mime_type = document.extension
        
        ext_map = {
            # Text-like docs
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain',
            # Presentations
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'ppt': 'application/vnd.ms-powerpoint',
            # Images
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
        }
        mime_type = ext_map.get(document.extension, 'application/octet-stream')
        
        text = extract_text_from_file(file_path, mime_type)
        if not text:
            logger.warning(f"No text extracted for doc {document.id}")
            return

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Generated {len(chunks)} chunks for doc {document.id}")

        # 3. Generate Embeddings & Save
        # NOTE:
        # - calling the embedding API per chunk for correctness and simplicity.
        objs = []
        for idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            # Call Gemini Embedding API
            # model="models/text-embedding-004" is standard efficient model (768 dim)
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk_text,
                task_type="retrieval_document",
            )

            # Support both dict-like and object-like responses and the ".values" attribute
            if isinstance(result, dict):
                embedding_obj = result.get("embedding")
            else:
                embedding_obj = getattr(result, "embedding", None)

            if embedding_obj is None:
                raise ValueError("No embedding returned from Gemini for a document chunk.")

            # Some client versions return an object with a `.values` attribute
            embedding = list(getattr(embedding_obj, "values", embedding_obj))

            objs.append(
                DocumentChunk(
                    document=document,
                    content=chunk_text,
                    embedding=embedding,
                    chunk_index=idx,
                )
            )

        if objs:
            DocumentChunk.objects.bulk_create(objs)
        
        document.is_processed = True
        document.save()
        logger.info(f"Successfully processed RAG for doc {document.id}")

    except Exception as e:
        logger.error(f"RAG processing failed: {e}", exc_info=True)