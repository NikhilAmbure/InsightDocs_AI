import requests
import tempfile
import os

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