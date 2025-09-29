import re
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncAzureOpenAI
from aiolimiter import AsyncLimiter
import os
from pathlib import Path
import base64
import ast

import tiktoken  # <-- added

# Setup rate limiters
request_limiter = AsyncLimiter(100, 60)     # 500 requests per minute
token_limiter = AsyncLimiter(100_000, 60)   # 500,000 tokens per minute

# Initialize the encoder for your model
encoder = tiktoken.encoding_for_model("gpt-4")

async def grade_questions(
    df_questions: pd.DataFrame,
    std_questions: pd.DataFrame,
    rubric: pd.DataFrame,
    openai_client: AsyncAzureOpenAI,
    model: str = "gpt-4",
    page_mapping: pd.DataFrame = None,
    img_dir: str = None,
    token_tracker= None
) -> pd.DataFrame:
    df_questions = df_questions[["original_file_name","submission_id", "question_number", "answer_text"]].copy()
    std_questions = std_questions[["question_number", "question_text", "question_context"]].copy()
    rubric = rubric[["question_number", "rubric", "total_points"]].copy()

    merged = pd.merge(std_questions, rubric, on="question_number", how="left")
    df_merged = pd.merge(df_questions, merged, on="question_number", how="left")
    results = []

    # Merge in the page mapping if provided
    if page_mapping is not None:
        df_merged = pd.merge(df_merged, page_mapping[["submission_id", "question_number", "pages"]], on=["submission_id", "question_number"], how="left")
    else:
        df_merged["pages"] = None

    async def grade_row(row):
        system_prompt = (
            "You are an AI grader specialized in question-level evaluation. You cannot evaluate links!"
            "Given the question details, rubric, student's answer, and images of their submission, grade strictly based on the rubric. "
             "Many questions require links that have been swapped out for the contents of that link via webscraping. If a question requires a link but has instead what appears to be a chatbot conversation, do not make any reference to the missing link."
             "If it appears the user tried to submit a link (for example: 'Link to AI bot: PingPong') however the hyperlink got lost due to converting text to markdown, set needs_human_eval to TRUE"
             "Generally, we have access to the content of any link of the regex form r'https://pingpong.hks.harvard.edu/group/d+/thread/(d+)', and try to swap those in prior to AI grading. However, sometimes student submit invalid PingPong links or hypertext as noted above. If an answer has a pingpong link (or any other link) but no further conversation context (we paste conversations in.) Set needs_human_eval to true. "
            "If the answer contains non-gradable content or rubric says human evaluation is required, set 'needs_human_eval' to TRUE. "
            "Return a JSON object inside triple backticks, no extra commentary."
        )

        # Parse images
        images = []
        image_tokens = 0
        image_refs = []
        if img_dir and row.get("pages"):
            try:
                pages = ast.literal_eval(row["pages"]) if isinstance(row["pages"], str) else row["pages"]
                if not isinstance(pages, list):
                    pages = [pages]
            except Exception:
                pages = [row["pages"]]
            for page_num in pages:
                img_name = f"{Path(row['original_file_name']).stem}_page_{int(page_num)}.png"
                img_path = os.path.join(img_dir, img_name)
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                        images.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                        # Estimate tokens for this image (adjust as needed)
                        image_tokens += round(17 * 22 * 1.7)
                        image_refs.append(img_name)

        # Build user prompt, referencing images
        user_text = f"""
QUESTION CONTEXT:
{row['question_context']}

QUESTION:
{row['question_text']}

STUDENT ANSWER:
{row['answer_text']}

RUBRIC:
{row['rubric']}

TOTAL POINTS: {row['total_points']}

Image(s) of the student's submission: {', '.join(image_refs) if image_refs else 'None'}

Please grade the student's answer. Return a JSON object inside triple backticks with:
- "points_awarded"
- "grade_explanation"
- "needs_human_eval"
"""

        # Token estimation
        encoding = tiktoken.get_encoding("o200k_base")
        system_tokens = len(encoding.encode(system_prompt))
        user_tokens = len(encoding.encode(user_text))
        total_tokens = system_tokens + user_tokens + image_tokens
        if token_tracker:
            token_tracker.add("grading", total_tokens)

        #print(f"Estimated tokens for submission_id={row['submission_id']}, question={row['question_number']}: {total_tokens}")

        # Build messages for OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}] + images},
        ]

        try:
            async with request_limiter:
                await token_limiter.acquire(total_tokens)
                response = await openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                )

            llm_output = response.choices[0].message.content

            pattern = r"```(?:json)?(.*?)```"
            match = re.search(pattern, llm_output, re.DOTALL)
            raw_json = match.group(1).strip() if match else llm_output.strip()

            data = json.loads(raw_json)
            if not isinstance(data, dict):
                raise ValueError("LLM output is not a JSON object.")

            data.update({
                "question_number": row["question_number"],
                "question_context": row["question_context"],
                "question_text": row["question_text"],
                "total_points": row["total_points"],
                "submission_id": row["submission_id"],
                "answer_text": row["answer_text"],
                "rubric": row["rubric"]
            })

            return data

        except Exception as e:
            print(f"Error grading submission_id={row['submission_id']}, question={row['question_number']}: {e}")
            return {
                "submission_id": row["submission_id"],
                "question_number": row["question_number"],
                "question_context": row['question_context'],
                "question_text": row["question_text"],
                "answer_text": row["answer_text"],
                "rubric": row["rubric"],
                "points_awarded": 0,
                "total_points": row["total_points"],
                "grade_explanation": "",
                "needs_human_eval": True
            }

    tasks = [grade_row(row) for _, row in df_merged.iterrows()]

    # Just await them all together — limiter handles the pacing
    for result in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Grading Questions"):
        res = await result
        results.append(res)

    if token_tracker:
        token_tracker.print_process("grading")
    
    return pd.DataFrame(results)


async def grade_questions_simple(
    df: pd.DataFrame,
    openai_client: AsyncAzureOpenAI,
    n: int,
    model: str = "gpt-4",
    bar_desc: str | None = None,
    token_tracker = None
) -> pd.DataFrame:
    """
    Adds a column `grade_{n}` with integer points (or pd.NA on failure).
    Displays a tqdm_asyncio progress bar.
    """

    bar_desc = bar_desc or f"Grading pass {n}"
    system_prompt = (
        "You are an AI grader.  Using the rubric, decide how many points "
        "to award (0 – TOTAL_POINTS).  Reply ONLY with a JSON object, e.g.\n"
        '{"points_awarded": 3}'
    )

    async def grade_row(idx: int, row):
        # build the user prompt
        user_prompt = f"""
QUESTION CONTEXT:
{row['question_context']}

QUESTION:
{row['question_text']}

STUDENT ANSWER:
{row['answer_text']}

RUBRIC:
{row['rubric']}

TOTAL_POINTS: {row['total_points']}
"""
        # count tokens
        prompt_tokens = (
            len(encoder.encode(system_prompt))
            + len(encoder.encode(user_prompt))
        )
        if token_tracker:
            token_tracker.add("fast_grading", prompt_tokens)
        # rate-limit
        async with request_limiter:
            await token_limiter.acquire(prompt_tokens)
            try:
                resp = await openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role":"system","content":system_prompt},
                        {"role":"user","content":user_prompt},
                    ]
                )
            except (Exception) as e:
                # API rejected us or network issue
                return idx, pd.NA
            # parse out the JSON object (Azure gives you a true dict here)
        try:
            content = resp.choices[0].message.content
            # content should already be a dict when using json_object:
            data = content if isinstance(content, dict) else json.loads(content)
            pts  = data.get("points_awarded", 0)
            return idx, pts
        except Exception as e:
            # log the very first bad payload so you can inspect it in your logs
            print(e)
            if not hasattr(grade_row, "_logged"):
                print("=== JSON PARSE FAILED ===\n", resp.choices[0].message.content)
                grade_row._logged = True
            return idx, pd.NA

    # launch & gather with a live tqdm bar
    tasks   = [grade_row(i, r) for i, r in df.iterrows()]
    results = await tqdm_asyncio.gather(*tasks, desc=bar_desc)
    # rebuild the grade list in order
    grades = [pd.NA] * len(results)
    for idx, pts in results:
        grades[idx] = pts

    df[f"grade_{n}"] = grades
    return df