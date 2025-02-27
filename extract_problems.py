import os
import asyncio
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

async def extract_questions_from_markdown(markdown_text: str, openai_client: AsyncAzureOpenAI, model = "gpt-4o") -> list:
    """
    Extract questions from markdown text using Azure OpenAI.
    Returns a list of dictionaries containing question numbers and text.
    """
    system_prompt = (
        "You are an AI assistant that extracts questions from markdown text. "
        "Analyze the provided markdown and return a JSON array of questions found, with format:\n"
        "[\n"
        "  {\n"
        "    \"question_number\": \"1\",\n" 
        "    \"question_text\": \"...\"\n"
        "  },\n"
        "  ...\n"
        "]\n"
        "Only include actual questions, numbered or unnumbered. No extra text or keys."
    )

    user_message = f"""
    Please extract all questions from this markdown text:

    {markdown_text}

    Return only the JSON array described above.
    """

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0
        )
        
        # Extract JSON from response
        result = response.choices[0].message.content
        # Clean up response to get just the JSON part
        json_str = result.strip().strip('```').strip()
        return json.loads(json_str)

    except Exception as e:
        print(f"Error extracting questions: {e}")
        return []

async def process_submissions(df: pd.DataFrame, output_csv: str = "extracted_questions.csv") -> pd.DataFrame:
    """
    Process submissions dataframe to extract questions from markdown content.
    
    Args:
        df: DataFrame with columns 'submission_id' and 'markdown'
        output_csv: Path to save backup CSV
        
    Returns:
        DataFrame with columns: submission_id, question_number, question_text
    """
    load_dotenv()
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT"),
        api_key=os.getenv("AZURE_API_KEY_GPT"),
        api_version="2024-02-15-preview"
    )

    results = []
    
    async def process_submission(row):
        questions = await extract_questions_from_markdown(row['markdown'], client)
        for q in questions:
            results.append({
                'submission_id': row['submission_id'],
                'question_number': q['question_number'],
                'question_text': q['question_text']
            })
    
    # Create tasks for each submission
    tasks = [process_submission(row) for _, row in df.iterrows()]
    
    # Run tasks with progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting Questions"):
        await coro
        
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # Save backup
    df_results.to_csv(output_csv, index=False)
    
    return df_results
