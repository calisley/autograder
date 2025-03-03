import os
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

#TODO: convert to o1 with image support upload
#TODO: feed images / pdfs just to o1, look at generated answers

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
        content = content.strip().strip('```').strip()

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
    output_csv: str = "./temp/standardized_answer_key_all_attempts.csv"
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


async def process_best_answer_for_question(
    question_num: str,
    group_df: pd.DataFrame,
    client: AsyncAzureOpenAI,
    model: str
) -> dict:
    """
    Helper function that takes all attempts for a single question, along with
    question metadata, and prompts an LLM to pick the best answer.

    :param question_num: The question_number (as a string or int).
    :param group_df: A subset of the merged dataframe containing all attempts
                     for this question_number.
    :param client: The AsyncAzureOpenAI client to call.
    :param model: The Azure OpenAI model name to use.
    :return: A dict with the best answer for this question.
    """

    # We'll assume all rows in group_df share the same question_context, question_text, points
    question_context = str(group_df.iloc[0].get("question_context", ""))
    question_text = str(group_df.iloc[0].get("question_text", ""))

    # Build a summary of all attempts
    # Each row has 'attempt_number', 'answer'
    attempts_summary = []
    for _, row in group_df.iterrows():
        att_num = row["attempt_number"]
        ans = row.get("answer", "")
        expl = row.get("explanation", "")
        attempts_summary.append(f"Attempt {att_num}:\nAnswer: {ans}\nExplanation: {expl}\n")

    attempts_str = "\n".join(attempts_summary)

    system_prompt = (
        "You are a specialized AI teacher. You will see multiple different answers "
        "to the same question. Your task is to determine which answer is the best, "
        "most correct, most complete response to that question, and most faithfully gets at the spirit of the question."
    )

    # Adjust the user prompt as needed for your use case
    user_prompt = f"""
    QUESTION CONTEXT:
    {question_context}

    QUESTION TEXT:
    {question_text}

    BELOW ARE ANSWERS FROM MULTIPLE ATTEMPTS. PICK THE BEST ONE:

    {attempts_str}

    This carefully about deciding on the best answer, and closely consider exactly what the question is asking. You should be faithful to the spirit of the question. 
    If a question asks for an approximmate answer, the exact answer is no better than a good approximation.
    As a default, assume the average answer provided is correct, but you may need to adjust this based on the specific question and answers provided.
    
    If the question is an open ended writing prompt, return a small sample of potentially valid answers, concatenated together in one string, titled "Possible responses:"
    If the question is a multiple choice question or true false question, and allows for an explanation, allow for some leeway in terms of reasoning as potential correct answers.
    Provide the best answer verbatim (or adapt as needed to ensure correctness),
    along with that answers's explanation or reasoning.
    Return it in the following JSON format only:

    NEVER explicitly reference a provided example answer as part of the answer key. 

    {{
      "best_answer": "...",
      "best_explanation": "..."
    }}
    """

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        content = content.strip().strip('```').strip()
        content = content.strip("json")
        try:
            json_obj = json.loads(content)
            best_answer = json_obj.get("best_answer", "")
            best_explanation = json_obj.get("best_explanation", "")
        except json.JSONDecodeError:
            # If the LLM did not strictly follow JSON instructions, fallback
            best_answer = content
            best_explanation = "Could not parse JSON. The raw LLM response is stored in best_answer."

        return {
            "question_number": question_num,
            "best_answer": best_answer,
            "best_explanation": best_explanation
        }
    except Exception as e:
        print(f"Error while picking best answer for question {question_num}: {e}")
        return {
            "question_number": question_num,
            "best_answer": "",
            "best_explanation": f"Error: {e}"
        }


async def select_best_responses(
    merged_df: pd.DataFrame,
    client = AsyncAzureOpenAI,
    model: str = "gpt-4o",
    output_csv: str = "./temp/standardized_answer_key.csv"
) -> pd.DataFrame:
    """
    Takes the merged DataFrame from `generate_key`, which contains multiple attempts
    per question_number, and queries the LLM to select the best attempt for each question.

    Returns a new DataFrame with columns:
    [
      'question_number',
      'best_answer',
      'best_explanation'
    ]
    (One row per question_number.)
    """
    # Group by question_number
    grouped = merged_df.groupby("question_number")

    # Build tasks for each question_number
    tasks = []
    for question_num, group in grouped:
        tasks.append(
            process_best_answer_for_question(question_num, group, client, model)
        )

    # Run tasks concurrently with a progress bar
    results = []
    for coro in tqdm_async(asyncio.as_completed(tasks), total=len(tasks), desc="Selecting Best Answers"):
        result = await coro
        results.append(result)

    # Convert to DataFrame
    best_answers_df = pd.DataFrame(results)

    # Return it. If needed, you can also merge back with `merged_df` on question_number.
    questions_with_best_answers = best_answers_df.merge(merged_df, on="question_number", how="left")
    columns_to_drop = ["answer", "explanation", "attempt_number"]
    questions_with_best_answers = questions_with_best_answers.drop(columns=columns_to_drop, errors="ignore")
    
    # Remove duplicate rows to return a single row per question_number
    questions_with_best_answers = questions_with_best_answers.drop_duplicates(subset=["question_number"])
    questions_with_best_answers.to_csv(output_csv, index=False)
    return questions_with_best_answers
