import re
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncAzureOpenAI
from aiolimiter import AsyncLimiter
import tiktoken

# Setup rate limiters
request_limiter = AsyncLimiter(100, 60)     # 500 requests per minute
token_limiter = AsyncLimiter(100_000, 60)   # 500,000 tokens per minute

async def generate_subquestion_feedback(
    df_feedback: pd.DataFrame,
    openai_client: AsyncAzureOpenAI,
    model: str = "gpt-4",
    token_tracker=None,
) -> pd.DataFrame:
    """
    For **each row** (i.e. each sub‑question) in ``df_feedback`` add a new column
    ``question_feedback`` containing short, casual AI feedback.

    Input ``df_feedback`` is assumed to have these columns exactly:
        • ``points_awarded``
        • ``grade_explanation``
        • ``needs_human_eval`` (bool‑like)
        • ``question_number``  (e.g. "1a")
        • ``total_points``
        • ``submission_id``
        • ``answer_text``
        • ``rubric``
    """
    # --------------------------------------------------------------
    # 1.  Prepare dataframe.
    # --------------------------------------------------------------
    df = df_feedback.copy()
    df["needs_human_eval"] = df["needs_human_eval"].astype(bool)
    df["question_feedback"] = ""

    # Initialize tiktoken encoder
    encoder = tiktoken.get_encoding("o200k_base")

    # --------------------------------------------------------------
    # 2.  System prompt (unchanged)
    # --------------------------------------------------------------
    SYSTEM_PROMPT =  """
You’re a chill older-Gen-Z teaching assistant. Keep feedback **very** short and only give it when there’s something meaningful to say.

WHEN TO GIVE FEEDBACK
• If the student earned a low score ➜ identify the main issue and one way to fix it.
• If the answer is missing, blank, obviously incomplete, or 'NaN' ➜ respond with “Missing or incomplete”.
• Otherwise unremarkable work, checkbox/ completion-only items, link-sharing confirmations, etc. ➜ give NO feedback.
• If the provided answer is 'nan' always return “Missing or incomplete”.
Never mention "the assignment/prompt criteria/requirement." nor praise people for "meeting the assignment criteria." Meeting criteria or requirements in the rubric/grade explanation is not worthy of feedback.
• For average work on questions requiring effort beyond completion, you should include a one of "Nice job" "Good work", "Excellent!", "Great!", etc. and at LEAST one specific highlight from their answer. For example "especially the detailed AI-integrity strategy."
    - Note that these 'specific highlights' should never be vague. Something like "The system prompt clearly guides the AI's behavior with detail and a friendly tone" is too broad, and doesn't highlight specifically how the student did well. Instead, you might say "The system prompt clearly guides the AI's behavior with detail and a friendly tone, especially the part about 'being chill' and 'keeping it short.'"
WHAT NEVER TO DO
• Never mention links, URLs, check-boxes, or the rubric—no matter what the question or grade explanation says. Even if the students are required to submit a link for a question and do not, never, EVER mention it in your feedback. 
• Do not praise link-sharing or form completion.
• Do not praise or mention anything graded on completion. 
Questions that could be answered via a checkbox (e.g. coding experience) should never recieve feedback. 
Questions with responses like
 "I have shared the link [ X] YES [ ] NO" 
Should NEVER RECEIVE FEEDBACK. 


Never mention links! EVER! "link works well" or "and the link is easily accessible." should NEVER be said. NEVER EVER COMMENT ON LINKS.

Dont say: 
"Nice job indicating completion. Keep it up!"
Instead, say:
"No feedback".
Don't say:
"Nice work including both an explanation and a link"
Instead say:
"Nice work!" with a short  comment on the quality of the explanation. 
Dont ay:
"Nice job—your transcript link and reflective reflection clearly met the criteria."
Instead say: "Nice job"


• Important: NEVER discuss links. 

• Avoid filler like “Great work!” unless you have already decided feedback is needed for a substantive reason.

TONE & LENGTH
• When giving feedback, you should provide 1–3 crisp sentences, max ~40 words.
- When not giving feedback, return either an empty string '', "No feedback indicated" or "Missing or incomplete" 
• Polite, direct, and specific; no jargon or gushy language.
• Vary phrasing (“Well done highlighting…”, “You might consider…”, “Another angle is…”).
• Offer concrete places for improvement, not a laundry list.
Positive feedback should highlight specific creative aspects of their responses. 

OUTPUT FORMAT
Return **only** this JSON, wrapped in triple back-ticks:

```json
{"question_feedback": "<your feedback here or empty string>"}
"""
    combined_system = SYSTEM_PROMPT
    system_tokens = len(encoder.encode(combined_system))

    # --------------------------------------------------------------
    # 3.  Process each row with rate limiting and exact token counts
    # --------------------------------------------------------------
    async def process_row(idx: int, row: pd.Series) -> tuple[int, str]:
        if row["needs_human_eval"]:
            return idx, "Awaiting human review."

        markdown = (
            f"### Sub‑question {row['question_number']}\n\n"
            f"### Question Context\n\n{row.get('question_context','')}\n\n"
            f"**Student Answer:**\n\n{row['answer_text']}\n\n"
            f"**Points Awarded:** {row['points_awarded']} / {row['total_points']}\n\n"
            f"**Grade Explanation:**\n\n{row['grade_explanation']}\n\n"
        )
        user_prompt = f"Please write casual feedback for the sub‑question below \n\n{markdown}"

        try:
            user_tokens = len(encoder.encode(user_prompt))
            total_tokens = system_tokens + user_tokens
            if token_tracker:
                token_tracker.add("feedback_generation", total_tokens)
            async with request_limiter:
                await token_limiter.acquire(total_tokens)
                response = await openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": combined_system},
                        {"role": "user", "content": user_prompt},
                    ],
                )

            content = response.choices[0].message.content
            match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            raw_json = match.group(1).strip() if match else content.strip()
            data = json.loads(raw_json)
            feedback = data.get("question_feedback", "")
            clean_feedback = await judge_and_clean_feedback(
                feedback,
                row["grade_explanation"],
                openai_client,
            )
            return idx, clean_feedback

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            return idx, ""

    # Launch all tasks concurrently without manual batching
    tasks = [process_row(idx, row) for idx, row in df.iterrows()]

    # Gather feedback with live progress bar
    for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating Feedback"):
        idx, feedback = await future
        df.at[idx, "question_feedback"] = feedback

    return df


async def judge_and_clean_feedback(
    feedback: str,
    grade_explanation: str,
    openai_client: AsyncAzureOpenAI,
    model: str = "gpt-4o",
    token_tracker=None,
) -> str:
    """
    Second-pass validator that enforces style/content rules.
    If the draft feedback breaks a rule, swap it for a bland
    ‘Good work.’ or ‘Nice job.’ plus one brief note on missing points.
    Always return {"question_feedback": …} JSON.
    """

    JUDGE_SYSTEM = """
Your task is to audit a short piece of TA feedback and ensure it **never**:
• Mentions links, URLs, check-boxes, or the rubric
• Praises link sharing / form completion
• Gives feedback on completion-only questions

If the feedback violates *any* rule, remove that part of the feedback and return the cleaned feedback. 

Otherwise, return the feedback exactly as you recieved it. Do not provide commentary on the feedback. Your job is to fix it or leave it as is. You are a judge, but do not ever return your judgement. 

Wrap the final result in triple-back-ticked JSON, exactly:
```json
{"question_feedback": "<clean feedback or empty string>"}
```

"""

    user_prompt = f"""Draft feedback:
    {feedback}
    """
    if token_tracker:
        token_tracker.add("feedback_judging", len(feedback))
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",  "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
    raw_json = match.group(1).strip() if match else content.strip()
    data = json.loads(raw_json)
    return data.get("question_feedback", "")