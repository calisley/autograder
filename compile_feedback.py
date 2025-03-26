import re
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncAzureOpenAI

async def generate_overall_feedback(
    df_feedback: pd.DataFrame,
    openai_client: AsyncAzureOpenAI,
    model: str = "gpt-4",
    batch_size: int = 10  # Batch size for concurrent requests
) -> pd.DataFrame:
    """
    Generates overall feedback by aggregating question-level grading feedback for each submission.
    
    Expects df_feedback to have columns:
      ["submission_id", "question_number", "answer_text", "points_awarded", "total_points", 
       "grade_explanation", "needs_human_eval", "rubric"]
    
    For each submission, the function compiles all question-level feedback into a Markdown prompt,
    then feeds that prompt to an LLM instructed to act as a thoughtful teaching assistant. The LLM
    returns overall, concise, and tangible feedback for the student.
    
    Returns a DataFrame with:
      submission_id, overall_feedback
    """
    # Group the feedback by submission_id
    submission_ids = df_feedback["submission_id"].unique()
    results = []
    
    async def grade_submission(submission_id: str) -> dict:
        # Filter the rows for this submission
        group = df_feedback[df_feedback["submission_id"] == submission_id]
        
        # Build the aggregated Markdown for all questions in this submission
        markdown = f"# Submission ID: {submission_id}\n\n"
        for _, row in group.iterrows():
            markdown += f"## Question {row['question_number']}\n\n"
            markdown += f"**Student Answer:**\n\n{row['answer_text']}\n\n"
            markdown += f"**Points Awarded:** {row['points_awarded']} / {row['total_points']}\n\n"
            markdown += f"**Grade Explanation:**\n\n{row['grade_explanation']}\n\n"
            if pd.notna(row['rubric']) and row['rubric'].strip():
                markdown += f"**Rubric:**\n\n{row['rubric']}\n\n"
            if pd.notna(row['needs_human_eval']) and row['needs_human_eval']:
                markdown += f"**Note:** This answer requires human evaluation.\n\n"
            markdown += "---\n\n"
        
        # Define system and user prompts
        system_prompt = """
         You are a thoughtful teaching assistant who has been given question-level feedback on a student's submission. Your task is to produce concise, non-repetitive, and helpful overall feedback for the student by summarizing points from each question’s “grade explanation.”

Some of the feedback mentions external links, images, or otherwise indicates a need for human evaluation. **If a question or answer depends on any external reference (including links, images, or figures) or is labeled “Extra Credit” without meaningful textual content, completely skip that item. Under no circumstance should you mention, critique, or even acknowledge external links, images, or Extra Credit in your final summary.** Similarly, avoid discussing the rubric or grading process.

Please follow these instructions:

1. **Combine multi-part questions:** If a question has parts (e.g., 1a, 1b), merge all relevant feedback into one concise summary for that question.  
2. **Skip external references:** Ignore or remove any feedback about links, figures, or images that require external access or human evaluation. Do not mention them at all in your final summary.  
3. **Skip or omit Extra Credit if it has no text-based answer:** If the student’s Extra Credit response is empty or simply references a link/image, exclude it entirely.  
4. **Focus on clarity and correctness of text-based content only:** Provide actionable insights for improvement based on the textual answers that can be evaluated.  
5. **Return your final response as valid JSON** enclosed in triple backticks, with exactly one key: `"overall_feedback"`.  
6. **Do not include any commentary** beyond the JSON object.



 """
        
        user_prompt = f"""
Please review the following aggregated feedback and student responses:

{markdown}

Provide your overall feedback:
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
            if not isinstance(data, dict) or "overall_feedback" not in data:
                raise ValueError("LLM did not return the expected JSON object with 'overall_feedback'.")
            
            return {
                "submission_id": submission_id,
                "overall_feedback": data["overall_feedback"]
            }
        except Exception as e:
            print(f"Error generating overall feedback for submission {submission_id}: {e}")
            return {
                "submission_id": submission_id,
                "overall_feedback": ""
            }
    
    # Create tasks for each submission
    tasks = [grade_submission(sub_id) for sub_id in submission_ids]
    
    # Process tasks in batches
    for i in tqdm_asyncio(range(0, len(tasks), batch_size), desc="Generating Overall Feedback"):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    
    return pd.DataFrame(results)
