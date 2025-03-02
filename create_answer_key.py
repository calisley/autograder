import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

#TODO: figure out how to standardize correct answer indication...
async def format_answer_key(markdown_text: str, openai_client: AsyncAzureOpenAI, model="gpt-4o") -> list:
    """
    Extract questions and answers from markdown text using Azure OpenAI.
    Returns a list of dictionaries containing:
      - question_number
      - question_text
      - points
      - answer_text
    """
    system_prompt = (
        "You are a strict JSON formatter. "
        "Do not include any text outside of the JSON. "
        "Only output valid JSON of the form:\n"
        "[\n"
        "  {\n"
        "    \"question_number\": \"1\",\n"
        "    \"question_text\": \"...\",\n"
        "    \"points\": 10,\n"
        "    \"provided_correct_answer\": \"...\"\n"
        "  },\n"
        "  ...\n"
        "]\n"
    )

    user_prompt = f"""
    Please extract all questions and their answers from the following markdown text. 
    Do NOT answer the questions or alter the answers provided. Simply structure the data. 
    There can be multiple correct answers to every question. If the answer key marks multiple answers as correct (EG: X TRUE T UNCERTAIN), return both answers. 
    Return the provided explanation as part of the answer. 
    
    {markdown_text}

    For each question, also extract the number of points it is worth.

    Return your answer as valid JSON only, no code fences.
    """

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0)
        
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


async def question_level_answer_key(
    markdown_text: str,
    output_csv: str = "./answer_key.csv"
) -> pd.DataFrame:
    """
    Extracts questions and answers from a single markdown string.

    Args:
        markdown_text: The markdown content as a string.
        output_csv: (Optional) File path to save the extracted data as CSV. Default: "extracted_questions.csv"

    Returns:
        A DataFrame with the columns:
            question_number, question_text, points, correct_answer
    """
    load_dotenv()
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT"),
        api_key=os.getenv("AZURE_API_KEY_GPT"),
        api_version="2024-12-01-preview"
    )

    # Extract question data from the markdown
    questions = await format_answer_key(markdown_text, client)

    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(questions)
    
    # Rename 'answer_text' to 'correct_answer' for clarity
    if "answer_text" in df_results.columns:
        df_results.rename(columns={"answer_text": "correct_answer"}, inplace=True)

    # Save to CSV if needed
    df_results.to_csv(output_csv, index=False)

    return df_results
