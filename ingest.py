import PyPDF2
import docx
from fastapi import UploadFile, HTTPException
import io
from typing import Callable, Dict

# Define a type alias for clarity
ExtractorFunc = Callable[[io.BytesIO], str]

# Define a dictionary of supported file types and their corresponding extractor functions
SUPPORTED_FILE_TYPES: Dict[str, ExtractorFunc] = {
    "application/pdf": lambda file: "\n".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda file: "\n".join(para.text for para in docx.Document(file).paragraphs),
    "text/plain": lambda file: file.read().decode("utf-8")
}

async def process_document(file: UploadFile) -> str:
    """
    Process the uploaded document and extract its text content.
    
    Args:
        file (UploadFile): The uploaded file object.
    
    Returns:
        str: The extracted text content from the document.
    
    Raises:
        HTTPException: If the file type is not supported.
    """
    content = await file.read()
    file_type = file.content_type

    if file_type not in SUPPORTED_FILE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file_type}")

    try:
        return SUPPORTED_FILE_TYPES[file_type](io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
