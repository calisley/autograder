import os
import asyncio
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

async def extract_questions_from_markdown(markdown_text: str, openai_client: AsyncAzureOpenAI, model="gpt-4o") -> list:
    """
    Extract questions and answers from markdown text using Azure OpenAI.
    Returns a list of dictionaries containing question_number, question_text, points, and answer_text.
    """
    system_prompt = (
        "You are a strict JSON formatter. "
        "Do not include any markdown or code fences in your response. "
        "Do not include any text outside of the JSON. "
        "Only output valid JSON of the form:\n"
        "[\n"
        "  {\n"
        "    \"question_number\": \"1\",\n"
        "    \"question_text\": \"...\",\n"
        "    \"points\": 10,\n"
        "    \"answer_text\": \"...\"\n"
        "  },\n"
        "  ...\n"
        "]\n"
    )

    user_prompt = f"""
    Please extract all questions and their answers from the following markdown text:

    {markdown_text}

    For each question, also extract the number of points it is worth. This information should be available in the question text.
    If points are not explicitly mentioned, set points to 0.

    Return your answer as valid JSON only, no code fences.
    """

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content

        # Attempt to parse as JSON
        json_str = result.strip().strip('```').strip()
        return json.loads(json_str)

    except json.JSONDecodeError:
        print("Failed to parse JSON. The response was:")
        print("-------- START --------")
        print(response.choices[0].message.content)
        print("-------- END   --------")
        return []
    except Exception as e:
        print(f"Other error: {e}")
        return []


async def process_submissions(df: pd.DataFrame, output_csv: str = "extracted_questions.csv") -> pd.DataFrame:
    """
    Process submissions dataframe to extract questions and answers from markdown content.
    
    Args:
        df: DataFrame with columns 'submission_id' and 'markdown'
        output_csv: Path to save backup CSV
        
    Returns:
        DataFrame with columns: submission_id, question_number, question_text, points, answer_text
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
                'question_text': q['question_text'],
                'points': q['points'],
                'answer_text': q['answer_text']
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
