import requests
import tempfile
import os
import logging 

from celery import shared_task
from .models import Document
from .utils.rag import process_document_for_rag

logger = logging.getLogger(__name__)

@shared_task
def process_document_task(document_id):

    try:
        document = Document.objects.get(id=document_id)

        try:
            file_path = document.file.path
        except NotImplementedError:
            file_path = document.file.url
        
        process_document_for_rag(document, file_path)

        logger.info(f"Task completed: Document {document_id} processed successfully")
    except Document.DoesNotExist:
        logger.info(f"Task failed: Document {document_id} not found.")
    except Exception as e:
        logger.error(f"Task failed for document {document_id}: {str(e)}")
        raise e

def prepare_local_document(document):
    """
    Downloads file from Cloudinary/URL to a local temp file.
    Returns: (path_to_file, cleanup_function)
    """
    # 1. Get URL
    file_url = document.file.url
    content = None

    # 2. Download or Read
    if file_url.startswith('http'):
        response = requests.get(file_url)
        content = response.content
    else:
        # Local file
        with document.file.open('rb') as f:
            content = f.read()

    # 3. Create Temp File
    ext = os.path.splitext(document.file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(content)
    tmp.close()

    # 4. Define cleanup
    def cleanup():
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    return tmp.name, cleanup