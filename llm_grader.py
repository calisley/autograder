import os
import re
import json
import asyncio
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import hashlib

def generate_question_id(question_text):
    """Generates a unique ID for a question using its text."""
    return hashlib.md5(question_text.encode()).hexdigest()[:10]  # Shorten hash to 10 characters

async def _grade_single_submission(
    submission_id: str,
    submission_markdown: str,
    answer_key_markdown: str,
    rubric_markdown: str,
    openai_client: AsyncAzureOpenAI,
    model: str = "gpt-4",
    #temperature: float = 0
) -> list:
    """
    Prompts the LLM to parse multiple questions from the answer key, find each
    corresponding answer in the student's Markdown, and return a JSON array:
      [
        {
          "question_text": "...",
          "student_answer": "...",
          "correct_answer": "...",
          "points_awarded": "...",
          "total_points": "...",
          "llm_explanation": "...",
          "needs_human_eval": true/false
        },
        ...
      ]
    """
    # 1) System prompt: the role / instructions
    system_prompt = (
        "You are an AI grader. You will:\n"
        "1) Parse the multiple questions and correct answers from the provided answer key.\n"
        "2) Locate the student's answers in their entire submission.\n"
        "3) Use the rubric to decide points_awarded. If the rubric is missing, use your discretion on the number of points to award. \n"
        "4) points_awarded should be included in the text of the question. Consider the answer key as ground truth, and compare the student's answer to it, as opposed to your own evaluation when assessing correctness.\n"
            "4a) both points_awarded and total_points should be singe numbers" 
        "5) If you cannot parse the student's answer (e.g., they circled both true and false for a true/false question, or selected multiple options on a single answer multiple choice) return True for needs_human_eval. Otherwise that field should always be false. \n"
        "5) Return a structured JSON array in triple backticks, with one object per question:\n\n"
        "[\n"
        "  {\n"
        "    \"question_text\": \"...\",\n"
        "    \"student_answer\": \"...\",\n"
        "    \"correct_answer\": \"...\",\n"
        "    \"points_awarded\": \"...\",\n"
        "    \"total_points\": \"...\",\n"
        "    \"llm_explanation\": \"...\"\n"
        "    \"needs_human_eval\": \"...\"\n"
        "  },\n"
        "  ...\n"
        "]\n"
        "No extra keys, no extra text."
    )

    # 2) User message: the actual documents
    user_message = f"""
    Below is the entire answer key (Markdown):

    {answer_key_markdown}


    Below is the student's submission (Markdown):

    {submission_markdown}


    Below is the grading rubric (Markdown) (if any):

    {rubric_markdown}


    **TASK**:
    - Extract each question from the answer key.
    - Find the student's corresponding answer within their submission.
    - Compare them, awarding points based on the rubric or your best interpretation.
    - Return valid JSON, inside triple backticks, with the structure described above.
"""

    try:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        }
        
        # if model == "o3-mini":
        #     params["reasoning_effort"] = "high"
            
        response = await openai_client.chat.completions.create(**params)
        llm_output = response.choices[0].message.content

        # Attempt to find JSON within triple backticks
        pattern = r"```(?:json)?(.*?)```"
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            raw_json = match.group(1).strip()
        else:
            # If no triple backticks, fallback to entire response
            raw_json = llm_output.strip()

        # Parse the JSON array
        data = json.loads(raw_json)

        # We expect a list of objects
        if not isinstance(data, list):
            data = [data]

        return data

    except Exception as e:
        print(f"[{submission_id}] Error grading submission: {e}")
        return []  # Return empty if error


async def _grade_all_submissions_async(
    df_submissions: pd.DataFrame,
    answer_key_markdown: str,
    rubric_markdown: str,
    model: str = "gpt-4",
    #temperature: float = 0
) -> pd.DataFrame:
    """
    Core async function that:
    - Initializes the Azure OpenAI client.
    - For each submission, calls _grade_single_submission.
    - Collects & flattens all results into a DataFrame.
    """

    load_dotenv()
    azure_endpoint = os.getenv("AZURE_ENDPOINT_GPT")
    azure_api_key = os.getenv("AZURE_API_KEY_GPT")

    if not azure_endpoint or not azure_api_key:
        raise ValueError("Missing Azure GPT credentials (AZURE_ENDPOINT_GPT, AZURE_API_KEY_GPT).")

    async with AsyncAzureOpenAI(
        api_key=azure_api_key,
        api_version="2024-12-01-preview",
        azure_endpoint=azure_endpoint,
    ) as openai_client:

        results = []

        async def handle_submission(row):
            submission_id = row["submission_id"]
            submission_markdown = row["markdown"]
            # Prompt the LLM
            json_array = await _grade_single_submission(
                submission_id=submission_id,
                submission_markdown=submission_markdown,
                answer_key_markdown=answer_key_markdown,
                rubric_markdown=rubric_markdown,
                openai_client=openai_client,
                model=model,
                #temperature=temperature
            )
            # Convert JSON objects to a list of dicts for final DF
            for index, obj in enumerate(json_array):
                # We'll store the submission_id with each record
                results.append({
                    "submission_id": submission_id,
                    "question_num": index+1,
                    "question_text": obj.get("question_text", "").strip(),
                    "student_answer": obj.get("student_answer", "").strip(),
                    "correct_answer": obj.get("correct_answer", "").strip(),
                    "points_awarded": obj.get("points_awarded", ""),
                    "total_points": obj.get("total_points", ""),
                    "llm_explanation": obj.get("llm_explanation", "").strip(),
                    "needs_human_eval": obj.get("needs_human_eval", "")
                })

        tasks = [handle_submission(row) for _, row in df_submissions.iterrows()]

        # Show progress bar while grading
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Grading Submissions"):
            await coro

    # Convert results to DataFrame
    df_result = pd.DataFrame(results, columns=[
        "submission_id",
        "question_num",
        "question_text",
        "student_answer",
        "correct_answer",
        "points_awarded",
        "total_points",
        "llm_explanation",
        "needs_human_eval"
    ])
    return df_result


async def grade_async(
    df_submissions: pd.DataFrame,
    answer_key_markdown: str,
    rubric_markdown: str = "",
    model: str = "gpt-4",
    #temperature: float = 0
) -> pd.DataFrame:
    """
    Synchronous wrapper that:
      - Takes a DataFrame of submissions with columns ["submission_id", "markdown"].
      - Takes entire assignment answer key (Markdown) and optional rubric (Markdown).
      - Calls an LLM to parse out all questions, awarding points for each question
        per submission.
      - Returns a DataFrame of:
        submission_id, question_num, question_text, student_answer, correct_answer,
        points_awarded, total_points, llm_explanation
    """
    return await _grade_all_submissions_async(
        df_submissions=df_submissions,
        answer_key_markdown=answer_key_markdown,
        rubric_markdown=rubric_markdown if rubric_markdown else "",
        model=model,
        #temperature=temperature
    )