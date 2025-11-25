import google.generativeai as genai
import time
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Constants
FILE_UPLOAD_TIMEOUT = 30  # seconds
API_RESPONSE_TIMEOUT = 60  # seconds
MAX_RETRIES = 3


def get_gemini_response(user_message, file_path, chat_history):
    """
    Get response from Gemini with document context.
    
    Args:
        user_message (str): User's current message
        file_path (str): Path to the document file
        chat_history (list): List of previous messages in format:
                            [{"role": "user"/"assistant", "content": "..."}]
    
    Returns:
        str: AI response text
    """
    try:
        logger.info(f"Starting Gemini response generation for: {user_message[:50]}")
        
        # 1. Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # 2. Upload file to Gemini
        logger.info(f"Uploading file: {file_path}")
        uploaded_file = upload_file_with_retry(file_path)
        
        if not uploaded_file:
            error_msg = "Could not process the document. Please ensure it's a valid PDF, DOCX, or text file."
            logger.error(error_msg)
            return error_msg

        logger.info(f"File uploaded successfully: {uploaded_file.name}")

        # 3. Build conversation history
        history = []
        
        # Add system context with uploaded file
        system_message = {
            "role": "user",
            "parts": [
                uploaded_file,
               (
                "You are a helpful AI assistant. "
                "First check if the user's question can be answered from the document. "
                "If yes — use the document and reference it directly. "
                "If no — intelligently use external knowledge to help. "
                "If the document has questions/exercises, solve them logically using both the document and your knowledge. "
                "Make answers clear, structured, and accurate."
                )

            ]
        }
        history.append(system_message)
        
        # AI acknowledgment
        history.append({
            "role": "model",
            "parts": ["I've read the document and I'm ready to help. What would you like to know?"]
        })
        
        # 4. Add previous chat history (skip empty messages)
        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "").strip()
            if content and content != "[Generating response...]":
                history.append({
                    "role": role,
                    "parts": [content]
                })
        
        # 5. Start chat and send current message
        logger.info(f"Starting chat session...")
        chat = model.start_chat(history=history)
        
        logger.info(f"Sending user message: {user_message[:50]}")
        response = chat.send_message(user_message)
        
        if not response or not response.text:
            error_msg = "No response from AI. Please try again."
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"Response received successfully: {response.text[:50]}")
        return response.text

    except Exception as e:
        error_type = type(e).__name__
        
        # Handle specific known exceptions
        if "BlockedPromptException" in error_type or "blocked" in str(e).lower():
            error_msg = "Your message was blocked by safety filters. Please rephrase your question."
            logger.warning(f"Blocked prompt: {str(e)}")
            return error_msg
        
        elif "APIError" in error_type or "api" in error_type.lower():
            error_msg = f"Gemini API Error: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
        elif isinstance(e, TimeoutError):
            error_msg = "Request timed out. The document might be too large. Please try again."
            logger.error(f"Timeout: {error_msg}")
            return error_msg
        
        else:
            error_msg = f"An error occurred. Please try again."
            logger.error(f"Unexpected error in get_gemini_response: {str(e)}", exc_info=True)
            return error_msg


def upload_file_with_retry(file_path, max_retries=MAX_RETRIES):
    """
    Upload file to Gemini with retry logic and polling.
    
    Args:
        file_path (str): Path to the file
        max_retries (int): Number of retry attempts
    
    Returns:
        genai.types.File or None: Uploaded file object or None if failed
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Upload attempt {attempt + 1}/{max_retries} for {file_path}")
            
            # Determine MIME type
            mime_type = get_mime_type(file_path)
            logger.info(f"Detected MIME type: {mime_type}")
            
            # Upload file
            uploaded_file = genai.upload_file(
                path=file_path,
                mime_type=mime_type
            )
            
            logger.info(f"File uploaded, waiting for processing. State: {uploaded_file.state.name}")
            
            # Poll until processing is complete with timeout
            start_time = time.time()
            while uploaded_file.state.name == "PROCESSING":
                elapsed = time.time() - start_time
                if elapsed > FILE_UPLOAD_TIMEOUT:
                    logger.error(f"File processing timeout after {elapsed:.1f}s")
                    return None
                
                logger.info(f"File processing... (elapsed: {elapsed:.1f}s)")
                time.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            # Check final state
            if uploaded_file.state.name == "ACTIVE":
                logger.info("File uploaded and processed successfully")
                return uploaded_file
            
            elif uploaded_file.state.name == "FAILED":
                logger.error(f"File upload failed. State: {uploaded_file.state}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
        
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Upload attempt {attempt + 1} failed ({error_type}): {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return None
    
    logger.error("All upload attempts failed")
    return None


def get_mime_type(file_path):
    """
    Determine MIME type based on file extension.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        str: MIME type
    """
    file_lower = file_path.lower()
    
    mime_map = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    
    for ext, mime_type in mime_map.items():
        if file_lower.endswith(ext):
            return mime_type
    
    return 'application/octet-stream'