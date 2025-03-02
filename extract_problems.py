import os
import asyncio
import pandas as pd
import json
from tqdm import tqdm
import aiofiles
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

async def structure_submissions(markdown_text: str, openai_client: AsyncAzureOpenAI, model="gpt-4o") -> list:
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
    Please create a structured representation of this assignment from the following markdown text:

    {markdown_text}

    In addition extract the number of points it is worth. This information should be available in the question text.
    Return your answer as valid JSON only. 
    """

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
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


async def process_submissions(df: pd.DataFrame, output_csv: str = "extracted_questions.csv", model: str ="gpt-4o") -> pd.DataFrame:
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
        api_version="2024-12-01-preview"
    )

    results = []
    
    async def process_submission(row):
        questions = await structure_submissions(row['markdown'], client, model)
        for q in questions:
            results.append({
                'submission_id': row['submission_id'],
                'question_number': q['question_number'],
                'question_text': q['question_text'],
                'answer_text': q['answer_text'],
                'points': q['points'],
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


async def get_questions_with_context(markdown_text: str, openai_client: AsyncAzureOpenAI, model="gpt-4o", output_csv:str = "./temp/questions_with_context.csv") -> list:
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
    )

    user_prompt = f"""
    Please create a structured representation of this assignment from the following markdown text:

    {markdown_text}

    In addition extract the number of points it is worth. This information should be available in the question text.

    question_context should be any information necessary to answer each question. question_context should only ever be text taken from the markdown. Do not add any of your own additional context. 
    - Questions will be viewed in isolation from one another. 
        - If you need to say "In the previous question..." or "In the figure below" instead explicitly provide all context for that question as context for the current question.  
    - NEVER POINT TO THE PREVIOUS QUESTION! The full context should be provided for every question that requires it.
    - Question context should include all information for figures relevant to solving the problem. 
    - In a series of questions that ask about the same set of information, the returned "question_context" should be identical for each question. 
    Return your answer as valid JSON only. 
    """

    try:
        response = await openai_client.chat.completions.create(
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













async def get_questions_with_context_2(
    df_extracted: pd.DataFrame,
    assignment_md: str,
    model: str = "gpt-4o"
) -> pd.DataFrame:
    """
    Given a DataFrame of extracted questions (df_extracted) and
    the original submissions DataFrame (df_submissions),
    this function calls an LLM asynchronously to incorporate relevant
    context (e.g., 'Figure 1' content) from each submission's markdown
    into the 'question_text'.
    """
    load_dotenv()
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT"),
        api_key=os.getenv("AZURE_API_KEY_GPT"),
        api_version="2024-12-01-preview"
    )


    system_prompt = (
        "You are an expert editor and educator that assists in deconstructing assignments. "
        "You will be given a question extracted from an exam or homework assignment, and a student's completed homework assignment. "
        "The question may be missing important context or figures lost in extraciton. Your job is to put that context back in place. "
        "You should ignore the student's answers to the question. The only text you should consider adding to questions comes from figures or context for previous questions."
        "You only return the updated question text."
        "IMPORTANT: You never solve the problem, nor relay the student's answer."
    )


    async def augment_row(row, markdown_text:str):
        """
        For each row, call the LLM to insert references or figure content
        from the markdown into the question_text if needed.
        """
        question_text = row['question_text']

        user_prompt = f"""
        The following question references certain material (like figures, tables, or other references) available in the markdown provided afterwards.
        Extract all information from the markdown necessary for solving the problem, and prepend it to the questiont text. 

        NEVER solve or answer the question. Your job is simply to reconstruct each question such that someone could solve it isolated from the other questions in the markdown document. 
        
        QUESTION:
        {question_text}
        
        MARKDOWN:
        {markdown_text}
        
        Return ONLY the updated question text. Do not provide any extra text.
        """

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            updated_question_text = response.choices[0].message.content.strip()
            return {
                'submission_id': row['submission_id'],
                'question_number': row['question_number'],
                'question_text': updated_question_text,
                'points': row['points'],
            }

        except Exception as e:
            # If anything fails, we return the original question text
            print(f"Error augmenting question for submission {row['submission_id']} question {row['question_number']}: {e}")
            return {
                'submission_id': row['submission_id'],
                'question_number': row['question_number'],
                'question_text': question_text,  # original
                'points': row['points'],
            }

    # Create and schedule one async task per row
    tasks = [
        augment_row(row, assignment_md)
        for _, row in df_extracted.iterrows()
    ]

    # Collect results with a progress bar
    updated_rows = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Augmenting Questions"):
        updated_rows.append(await coro)

    # Convert to DataFrame
    df_updated = pd.DataFrame(updated_rows)
    return df_updated



async def strip_assignment(
    df_submissions: pd.DataFrame,
    model: str = "gpt-4o",
    output_md: str= "./temp/blank_assignment.md"
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
    load_dotenv()
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT"),
        api_key=os.getenv("AZURE_API_KEY_GPT"),
        api_version="2024-12-01-preview"
    )

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

    return stripped_assignment