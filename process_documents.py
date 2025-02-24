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

async def analyze_document_async(file_path, document_client):
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
        )
        result = await poller.result()
        return result.content  # Extracted markdown text

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

async def process_all_documents_async(input_dir, markdown_dataframe, backup_dir=None):
    """
    Process all supported document types asynchronously and save extracted markdown text.

    Args:
        input_dir (str): Directory containing the documents.
        markdown_dataframe (str): CSV file path to save results.
        backup_dir (str, optional): Directory to save markdown backups.

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

        async def handle_file(file_name):
            """Processes a single file asynchronously."""
            file_path = os.path.join(input_dir, file_name)
            markdown_content = await analyze_document_async(file_path, document_client)
            
            if markdown_content:
                # Save backup as markdown file if required
                if backup_dir:
                    backup_filename = os.path.splitext(file_name)[0] + ".md"
                    backup_path = os.path.join(backup_dir, backup_filename)
                    async with aiofiles.open(backup_path, "w", encoding="utf-8") as md_file:
                        await md_file.write(markdown_content)
                
                results.append({"file_name": file_name, "markdown": markdown_content})

        # Create async tasks for each document
        tasks = [handle_file(f) for f in files]

        # Run all tasks with progress bar
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing documents"):
            await coro

        # Convert results to DataFrame and save
        df = pd.DataFrame(results, columns=["file_name", "markdown"])
        df.to_csv(markdown_dataframe, index=False)
        
        return df

# Function to run the async processing from a synchronous script
def run_document_processing(input_dir, markdown_dataframe, backup_dir=None):
    """
    Synchronous wrapper to run the async document processing function.

    Args:
        input_dir (str): Directory containing the documents.
        markdown_dataframe (str): CSV file path to save results.
        backup_dir (str, optional): Directory to save markdown backups.

    Returns:
        pd.DataFrame: The extracted markdown data.
    """
    return asyncio.run(process_all_documents_async(input_dir, markdown_dataframe, backup_dir))
