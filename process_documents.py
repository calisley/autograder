import os
import asyncio
import pandas as pd
import aiofiles  # Async file handling
from tqdm import tqdm
from dotenv import load_dotenv
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import DocumentContentFormat

# Supported document formats
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"}

async def analyze_document(file_path, document_client):
    """Asynchronously calls Azure Document Intelligence to analyze a document."""
    try:
        # Open file asynchronously using aiofiles
        async with aiofiles.open(file_path, "rb") as f:
            file_data = await f.read()  # Read file contents asynchronously

        # Start analysis
        poller = await document_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=file_data,
            output_content_format=DocumentContentFormat.MARKDOWN,
            polling=True,  # Ensure polling behavior is set
        )
        result = await poller.result()
        return result.content  # Extracted markdown text

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

async def process_all_documents(input_dir, markdown_dataframe=None, backup_dir=None, truncate=None):
    """
    Process all supported document types asynchronously and save extracted markdown text.

    Args:
        input_dir (str): Directory containing the documents.
        markdown_dataframe (str): CSV file path to save results.
        backup_dir (str, optional): Directory to save markdown backups.
        truncate (list[int], optional): List of page numbers to exclude from PDFs.

    Returns:
        pd.DataFrame: A DataFrame containing filenames and extracted markdown text.
    """
    load_dotenv()
    # Retrieve Azure credentials from environment variables
    AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
    AZURE_API_KEY = os.environ.get("AZURE_API_KEY")

    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        raise ValueError("Azure credentials (AZURE_ENDPOINT, AZURE_API_KEY) are missing from environment variables.")

    # Use async with to ensure proper cleanup of the Azure client
    async with DocumentIntelligenceClient(
        endpoint=AZURE_ENDPOINT, 
        credential=AzureKeyCredential(AZURE_API_KEY)
    ) as document_client:
        
        # Gather all supported document files
        files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            and os.path.isfile(os.path.join(input_dir, f))
        ]

        # Ensure backup directory exists if provided
        if backup_dir and not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)

        results = []
        batch_size = 10

        async def handle_file(file_name):
            """Processes a single file asynchronously."""
            file_path = os.path.join(input_dir, file_name)
            
            # If file is PDF and truncate is specified, handle page removal
            if file_name.lower().endswith('.pdf') and truncate:
                import PyPDF2
                
                # Read PDF and get number of pages
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    total_pages = len(pdf_reader.pages)
                    
                    # Create new PDF writer
                    pdf_writer = PyPDF2.PdfWriter()
                    
                    # Add all pages except those in truncate list
                    for page_num in range(total_pages):
                        if page_num + 1 not in truncate:  # page_num is 0-based, truncate list is 1-based
                            pdf_writer.add_page(pdf_reader.pages[page_num])
                    
                    # Save modified PDF to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        pdf_writer.write(temp_file)
                        temp_path = temp_file.name
                
                # Use temporary file for analysis
                markdown_content = await analyze_document(temp_path, document_client)
                # Clean up temporary file
                os.unlink(temp_path)
            else:
                markdown_content = await analyze_document(file_path, document_client)
            
            if markdown_content:
                # Save backup as markdown file if required
                if backup_dir:
                    backup_filename = os.path.splitext(file_name)[0] + ".md"
                    backup_path = os.path.join(backup_dir, backup_filename)
                    async with aiofiles.open(backup_path, "w", encoding="utf-8") as md_file:
                        await md_file.write(markdown_content)
                
                # Strip extension to create submission_id
                submission_id = os.path.splitext(file_name)[0]
                results.append({
                    "file_name": file_name,
                    "submission_id": submission_id,
                    "markdown": markdown_content
                })

        # Process files in batches of 10
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_tasks = [handle_file(f) for f in batch]
            
            # Run batch tasks with progress bar
            for coro in tqdm(asyncio.as_completed(batch_tasks), 
                           total=len(batch_tasks), 
                           desc=f"Processing batch {i//batch_size + 1}/{(len(files)-1)//batch_size + 1}"):
                await coro

        # Convert results to DataFrame and save
        df = pd.DataFrame(results, columns=["file_name", "submission_id", "markdown"])
        df.to_csv(markdown_dataframe, index=False)
        
        return df

# Function to run the async processing from a synchronous script
def run_document_processing(input_dir, markdown_dataframe, backup_dir=None, truncate=None):
    """
    Synchronous wrapper to run the async document processing function.

    Args:
        input_dir (str): Directory containing the documents.
        markdown_dataframe (str): CSV file path to save results.
        backup_dir (str, optional): Directory to save markdown backups.

    Returns:
        pd.DataFrame: The extracted markdown data.
    """
    return asyncio.run(process_all_documents(input_dir, markdown_dataframe, backup_dir, truncate))

async def process_single_document(file_path, output_md_path=None):
    """
    Asynchronously processes a single document and returns the markdown content.
    Optionally saves the markdown content to a file.

    Args:
        file_path (str): Path to the document file.
        output_md_path (str, optional): Path to save the generated markdown file.

    Returns:
        str: Extracted markdown content.
    """
    load_dotenv()
    # Retrieve Azure credentials from environment variables
    AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
    AZURE_API_KEY = os.environ.get("AZURE_API_KEY")

    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        raise ValueError("Azure credentials (AZURE_ENDPOINT, AZURE_API_KEY) are missing from environment variables.")

    # Use async with to ensure proper cleanup of the Azure client
    async with DocumentIntelligenceClient(
        endpoint=AZURE_ENDPOINT, 
        credential=AzureKeyCredential(AZURE_API_KEY)
    ) as client:
        markdown_content = await analyze_document(file_path, client)
        
        # Save markdown content to file if output path is provided
        if markdown_content and output_md_path:
            # Ensure directory exists
            output_dir = os.path.dirname(output_md_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Write markdown content to file
            async with aiofiles.open(output_md_path, "w", encoding="utf-8") as md_file:
                await md_file.write(markdown_content)
                
        return markdown_content

def process_single_document_sync(file_path, output_md_path=None):
    """
    Synchronous wrapper to run async processing for a single document.

    Args:
        file_path (str): Path to the document file.
        output_md_path (str, optional): Path to save the generated markdown file.

    Returns:
        str: Extracted markdown content.
    """
    return asyncio.run(process_single_document(file_path, output_md_path))
