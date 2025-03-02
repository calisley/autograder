import os
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI


async def generate_answer_and_explanation_json(
    question_text: str,
    question_context: str,
    openai_client: AsyncAzureOpenAI,
    model: str = "gpt-4o",
) -> dict:
    """
    Generate an answer and explanation for a given question using an LLM.

    The model is instructed to return a valid JSON object containing:
    {
      "answer": "...",
      "explanation": "..."
    }
    """
    system_prompt = (
        "You are an expert problem solver and educator. "
        "Please respond in valid JSON format only, with keys 'answer' and 'explanation'. "
        "Do NOT include any additional keys or text outside the JSON."
    )
    
    user_prompt = f"""
    QUESTION CONTEXT:

    {question_context}

    QUESTION:

    {question_text}

    Please return the result as a valid JSON object with exactly two keys:
    "answer" for the concise final answer, and
    "explanation" for a step-by-step or conceptual explanation.

    Example (generic):
    {{
      "answer": "answer (correct selection for selection questions, full response for open response)",
      "explanation": "some explanation, if appropriate"
    }}
    """

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            parsed_json = json.loads(content)
            if "answer" not in parsed_json or "explanation" not in parsed_json:
                raise ValueError("JSON does not contain 'answer' or 'explanation' keys.")
            return parsed_json
        except json.JSONDecodeError as decode_err:
            # Fallback if the JSON isn't valid
            return {
                "answer": "Failed to parse JSON from model response",
                "explanation": f"Raw content:\n{content}\nError:\n{decode_err}"
            }
    
    except Exception as e:
        return {
            "answer": "Failed to generate answer",
            "explanation": str(e)
        }


async def process_question_attempt(
    row: pd.Series,
    attempt_number: int,
    client: AsyncAzureOpenAI,
    model: str
) -> dict:
    """
    Helper function to process a single question attempt.
    Returns a dict with question_number, points, attempt_number, answer, and explanation.
    """
    question_text = row["question_text"]
    question_number = row["question_number"]
    question_context = row["question_context"]

    solution_json = await generate_answer_and_explanation_json(
        question_text=question_text,
        question_context=question_context,
        openai_client=client,
        model=model,
    )
    
    return {
        "question_number": question_number,
        "attempt_number": attempt_number,
        "answer": solution_json.get("answer", ""),
        "explanation": solution_json.get("explanation", "")
    }


async def generate_key(
    df: pd.DataFrame,
    n_attempts: int = 3,
    model: str = "gpt-4o",
    output_csv: str = "./temp/standardized_answer_key.csv"
) -> pd.DataFrame:
    """
    Given a DataFrame with columns:
        - 'question_context'
        - 'question_text'
        - 'question_number'
    
    Prompt the LLM to solve each question `n_attempts` times, returning
    a new DataFrame with columns:
        - 'question_number'
        - 'attempt_number'
        - 'answer'
        - 'explanation'
    """
    load_dotenv()

    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT"),
        api_key=os.getenv("AZURE_API_KEY_GPT"),
        api_version="2024-12-01-preview"
    )

    # 1. Build a list of coroutines (tasks) for all questions and attempts.
    tasks = []
    for _, row in df.iterrows():
        for attempt_num in range(1, n_attempts + 1):
            tasks.append(
                process_question_attempt(row, attempt_num, client, model)
            )

    # 2. Run all tasks concurrently, with a progress bar.
    results = []
    for coro in tqdm_async(asyncio.as_completed(tasks), total=len(tasks), desc="Generating solutions"):
        result = await coro
        results.append(result)


    # 3. Create a DataFrame from results
    results_df = pd.DataFrame(results)

    merged_df = results_df.merge(df, on="question_number", how="left")

    merged_df.to_csv(output_csv, index=False)
    
    return merged_df
