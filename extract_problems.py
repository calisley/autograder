import pandas as pd
import json
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncAzureOpenAI
import re
import tiktoken
from typing import Any

# async def structure_submissions(markdown_text: str, stripped_assignment:str, openai_client: AsyncAzureOpenAI, model="gpt-4o") -> list:
#     """
#     Extract questions and answers from markdown text using Azure OpenAI.
#     Returns a list of dictionaries containing question_number, question_text, points, and answer_text.
#     """
#     system_prompt = (
#         "You are a strict JSON formatter. "
#         "Do not include any markdown or code fences in your response. "
#         "Do not include any text outside of the JSON. "
#         "Only output valid JSON of the form:\n"
#         "[\n"
#         "  {\n"
#         "    \"question_number\": \"1\",\n"
#         "    \"question_text\": \"...\",\n"
#         "    \"points\": 10,\n"
#         "    \"answer_text\": \"...\"\n"
#         "  },\n"
#         "  ...\n"
#         "]\n"
#     )

#     user_prompt = f"""
#     Please create a structured representation of this assignment from the following markdown text:

#     {markdown_text}

#     In addition extract the number of points it is worth. This information should be available in the question text. 
#     Return your answer as valid JSON only. 
#     IMPORTANT: Questions may have multiple parts. Each part should be considered a separate question.
#     IMPORTANT: You must extract the entire student answer to each question as answer

#     For example, "1a", "1b", "1c" should be considered separate questions, labeled with their own question number.
#     """

#     try:
#         response = await openai_client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ]
#         )
        
#         result = response.choices[0].message.content

#         # Attempt to parse as JSON
#         json_str = result.strip().strip('```').strip()
#         return json.loads(json_str)

#     except json.JSONDecodeError:
#         print("Failed to parse JSON. The response was:")
#         print("-------- START --------")
#         print(response.choices[0].message.content)
#         print("-------- END   --------")
#         return []
#     except Exception as e:
#         print(f"Other error: {e}")
#         return []
async def process_submissions(
    df: pd.DataFrame,
    questions: pd.DataFrame,
    client: AsyncAzureOpenAI,
    output_csv: str = "extracted_answers.csv",
    model: str = "gpt-4o",
    token_tracker=None
) -> pd.DataFrame:
    """
    Process submissions to extract each student's answer for each question from markdown content,
    and append the corresponding question_context (looked up by question_number).

    Expects:
      - df: DataFrame with columns ["submission_id", "original_file_name", "markdown"]
      - questions: DataFrame with columns ["question_number", "question_text", "question_context", "points"]
      - output_csv: Path to save backup CSV

    Returns a DataFrame with columns:
      submission_id, original_file_name, question_number, question_text,
      question_context, answer_text, points
    """

    results: list[dict[str, Any]] = []

    async def process_submission(row: pd.Series):
        submission_id = row["submission_id"]
        submission_markdown = row["markdown"]
        file_name = row["original_file_name"]

        # Build a prompt listing all questions (number + text)
        questions_details = ""
        for _, question in questions.iterrows():
            questions_details += (
                f"\nQuestion Number: {question['question_number']}\n"
                f"Question: {question['question_text']}\n"
                "-----\n"
            )

        system_prompt = (
            "You are a strict JSON formatter. Do not include any markdown or code fences in your response. "
            "Do not include any text outside of the JSON. Only output valid JSON in the following format:\n"
            "[\n"
            "  {\n"
            "    \"question_number\": \"1\",\n"
            "    \"question_text\": \"...\",\n"
            "    \"answer_text\": \"...\"\n"
            "  },\n"
            "  ...\n"
            "]\n"
            "Return all question_number values in the format: 1a, 1b, 2a. If there are no subquestions, just return 1, 2, etc. "
            "Never return formats like 1.1, 1(1), 1-1, or 1 1; convert them to 1a, 1b, etc."
        )

        user_prompt = f"""
Please extract the student's answer for each question from the submission below.

Submission markdown:
{submission_markdown}

Below are the questions with their details:
{questions_details}

Return your answer as valid JSON only, in the following format:
[
  {{
    "question_number": "<question_number>",
    "question_text": "<question_text>",
    "answer_text": "<extracted answer>"
  }},
  ...
]
IMPORTANT: Do not include any additional commentary. Your response MUST be in valid JSON. Provide the answer to each question in the submission, do not cut off your response early.
"""
        if token_tracker:
            token_tracker.add("process_submissions", system_prompt+user_prompt)

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            result = response.choices[0].message.content
            pattern = r"```(?:json)?\s*(.*?)\s*```"
            match = re.search(pattern, result, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = result.strip()

            if not json_str:
                raise ValueError("Extracted JSON string is empty.")

            data = json.loads(json_str)

            # For each extracted answer, look up question_context and points, then append
            for obj in data:
                q_num = obj.get("question_number", "")
                # Find matching row in questions DataFrame
                question_row = questions.loc[questions["question_number"] == q_num]
                if not question_row.empty:
                    context = question_row.iloc[0]["question_context"]
                    points = question_row.iloc[0].get("points", None)
                else:
                    context = ""
                    points = None

                obj["question_context"] = context
                obj["points"] = points
                obj["original_file_name"] = file_name
                obj["submission_id"] = submission_id
                results.append(obj)

        except json.JSONDecodeError:
            print(f"Failed to parse JSON for submission {submission_id}.")
            print("Response was:")
            print(result)
            print(len(result))
        except Exception as e:
            print(f"Error processing submission {submission_id}: {e}")

    # Create one task per submission
    tasks = [process_submission(row) for _, row in df.iterrows()]

    # Process all tasks concurrently with a progress bar
    await tqdm_asyncio.gather(*tasks, desc="Processing Submissions")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

    if token_tracker:
        token_tracker.print_process("process_submissions")
    return df_results


async def get_questions_with_context(raw_assignment, client, model, output_csv, token_tracker=None):
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
        "    \"question_context\": \"1\",\n"
        "    \"question_text\": \"...\",\n"
        "    \"points\": 10,\n"
        "  },\n"
        "  ...\n"
        "]\n"
        "Return all question_number values in the format: 1a, 1b, 2a. If there are no subquestions, just return 1, 2, etc."
        "Never return formats like 1.1, 1(1), 1-1, or 1 1; convert them to 1a, 1b, etc."
    )

    user_prompt = f"""
    Please create a structured representation of this assignment from the following markdown text:

    {raw_assignment}

    In addition extract the number of points it is worth. This information should be available in the question text. If a question has multiple parts, and only specifies the point total for the whole question, use your discretion to divide the points among the parts.

    IMPORTANT: Questions may have multiple parts. Each part should be considered a separate question.

    question_context should be any information necessary to answer each question. question_context should only ever be text taken from the markdown. Do not add any of your own additional context. 
    - Questions will be viewed in isolation from one another. 
        - If you need to say "In the previous question..." or "In the figure below" instead explicitly provide all context for that question as context for the current question.  
    - NEVER POINT TO THE PREVIOUS QUESTION! The full context should be provided for every question that requires it.
    - Question context should include all information for figures relevant to solving the problem. 
    - In a series of questions that ask about the same set of information, the returned "question_context" should be identical for each question. 
    
    IMPORTANT: For questions with multiple parts, again they should each be returned as their own row, they should all contain the same context, unless additional context is added between parts. 
    For example, If we have [context] 1a, 1b, and [new context] 1c, you should return {{context}} for 1a and 1b, and {{context, new context}} for 1c, all as seperate rows.
    
    Return your answer as valid JSON only. 
    """

    if token_tracker:
        token_tracker.add("get_questions_with_context", system_prompt+user_prompt)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
         # The response should be raw JSON. Strip extra characters or code fences if present.
        json_str = response.choices[0].message.content.strip().strip('```').strip()

        # Parse the JSON string into a Python object (list of dicts).
        data = json.loads(json_str)

        # Build list of results to ensure we have consistent keys
        results = []
        for q in data:
            results.append({
                'question_number': q.get('question_number', ''),
                'question_context': q.get('question_context', ''),
                'question_text': q.get('question_text', ''),
                'points': q.get('points', '')
            })

        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Save backup to CSV
        df_results.to_csv(output_csv, index=False)

        if token_tracker:
            token_tracker.print_process("get_questions_with_context")
        return df_results
    
    except json.JSONDecodeError:
        print("Failed to parse JSON. The response was:")
        print("-------- START --------")
        print(response.choices[0].message.content)
        print("-------- END   --------")
        return []
    except Exception as e:
        print(f"Other error: {e}")
        return []

async def strip_assignment(
    df_submissions: pd.DataFrame,
    model: str = "gpt-4o",
    client:AsyncAzureOpenAI= None,
    output_md: str= "./temp/blank_assignment.md",
    token_tracker = None
) -> str:
    """
    Takes the first submission in df_submissions, sends its markdown to the LLM, 
    and asks the LLM to remove any student answers. The result is meant to 
    resemble the original assignment before completion.

    :param df_submissions: A DataFrame of submissions. 
        Must have at least a 'markdown_text' column.
    :param openai_client: An instance of AsyncAzureOpenAI for making chat completion requests.
    :param model: The name of the Azure OpenAI model to use.
    :return: A string containing the "stripped" assignment.
    """

    if len(df_submissions) == 0:
        raise ValueError("df_submissions is empty. Unable to strip assignment.")

    # Get the markdown from the first submission
    markdown_text = df_submissions.iloc[0]['markdown']

    # Construct prompts
    system_prompt = (
        "You are an assistant that removes any and all student-provided answers "
        "from an markdown version of assignment, restoring it to its original blank state."
        "Return only one string that faithfully recreates the original document. "
        "IMPORTANT: Do not remove any text critical to solving the problems, only the students' answers."
    )

    user_prompt = f"""
    Below is an assignment with a student's answers filled in. 
    Please remove all student answers or edits so that the document 
    is returned to its original, blank assignment form. 
    Keep any questions, prompts, figures, context, or instructions from the original assignment 
    exactly as they were.

    ASSIGNMENT WITH ANSWERS:
    {markdown_text}
    """
   
    if token_tracker:
        token_tracker.add("strip_assignment", system_prompt+user_prompt)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        stripped_assignment = response.choices[0].message.content.strip()
       

    except Exception as e:
        print(f"Error while stripping assignment: {e}")
        return markdown_text  # Fallback to original text in case of error
    
    async with aiofiles.open(output_md, "w", encoding="utf-8") as md_file:
        await md_file.write(stripped_assignment)

    if token_tracker:
        token_tracker.print_process("strip_assignment")
    return stripped_assignment