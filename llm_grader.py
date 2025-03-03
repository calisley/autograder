import re
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncAzureOpenAI


async def grade_questions(
    df_questions: pd.DataFrame,
    answer_key: pd.DataFrame,
    rubric: pd.DataFrame,
    openai_client: AsyncAzureOpenAI,
    model: str = "gpt-4",
    batch_size: int = 10  # Batch size for concurrent requests
) -> pd.DataFrame:
    """
    Grades question-level answers by merging student responses with the answer key and rubric.
    
    Expects:
      - df_questions: DataFrame with columns ["submission_id", "question_number", "answer_text"]
      - answer_key: DataFrame with columns ["question_number", "question_text", "question_context", 
                                            "best_answer", "best_explanation", "points"]
      - rubric: DataFrame with columns ["question_number", "rubric"] (Markdown)
    
    Returns a DataFrame with:
      question_number, points_awarded, total_points, grade_explanation, needs_human_eval
    """
    df_questions = df_questions[["submission_id", "question_number", "answer_text"]].copy()
    answer_key = answer_key[["question_number", "question_text", "question_context", "best_answer", "best_explanation", "points"]].copy()
    rubric = rubric[["question_number", "rubric"]].copy()
    
    # Merge answer key and rubric on question_number
    merged = pd.merge(answer_key, rubric, on="question_number", how="left")
    # Merge with student question-level answers
    df_merged = pd.merge(df_questions, merged, on="question_number", how="left")
    results = []
    
    async def grade_row(row):
        system_prompt = (
            "You are an AI grader specialized in question-level evaluation. You cannot evalutate links or images that cannot be converted to text."
            "Given the question details, rubric, and the student's answer, evaluate the student's response by awarding points strictly based on the rubric. "
            "IMPORTANT: If the answer contains something beyond your current capabilities (e.g. reading content that can't be translated to markdown, following a link, etc.), ALWAYS set 'needs_human_eval' to TRUE."
                "- Never assume the student did something in a figure or at an external link you cannot see." 
                "- IMPORTANT: If the rubric mentions requring human evaluation, set 'needs_human_eval' to TRUE."
            "Otherwise, if you cannot determine what grade to award based on the rubric, set 'needs_human_eval' to TRUE. "
            "You only return JSON objects in triple backticks, and never provide additional commentary. "
            
        )
        
        user_prompt = f"""        
        QUESTION CONTEXT:
        {row['question_context']}
        
        QUESTION:
        {row['question_text']}
        
        STUDENT ANSWER:
        {row['answer_text']}
        
        SUGGESTED ANSWER:
        {row['best_answer']}
        
        SUGGESTED EXPLANATION:
        {row['best_explanation']}
        
        RUBRIC:
        {row['rubric']}
        
        TOTAL POINTS: {row['points']}
        
        Please grade the student's answer. Return a JSON object in triple backticks with the following keys:
        - "points_awarded": a number representing the points awarded.
        - "grade_explanation": a brief explanation of the grade.
        - "needs_human_eval": a boolean indicating if the answer requires human evaluation.
        Do not include any additional commentary.
        """
        
        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            }
            response = await openai_client.chat.completions.create(**params)
            llm_output = response.choices[0].message.content
            
            # Extract JSON from triple backticks
            pattern = r"```(?:json)?(.*?)```"
            match = re.search(pattern, llm_output, re.DOTALL)
            if match:
                raw_json = match.group(1).strip()
            else:
                raw_json = llm_output.strip()
            
            data = json.loads(raw_json)
            if not isinstance(data, dict):
                raise ValueError("LLM did not return a JSON object.")
            data["question_number"] = row["question_number"]
            data["total_points"] = row["points"]
            data["submission_id"] = row["submission_id"]
            data["answer_text"] = row["answer_text"]
            data["rubric"] = row["rubric"]

            return data
        except Exception as e:
            print(f"Error grading question {row['question_number']}: {e}")
            return {
                "submission_id": row["submission_id"],
                "question_number": row["question_number"],
                "answer_text": row["answer_text"],
                "rubric": row["rubric"],
                "points_awarded": 0,
                "total_points": row["points"],
                "grade_explanation": "",
                "needs_human_eval": True
            }
    
    # Prepare tasks
    tasks = [grade_row(row) for _, row in df_merged.iterrows()]
    
    # Process tasks in batches
    for i in tqdm_asyncio(range(0, len(tasks), batch_size), desc="Grading Questions"):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        
    return pd.DataFrame(results)