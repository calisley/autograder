import asyncio
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
import json
from config import config

# --------------------------------------------------------------
# 1. Constants and semaphores for Azure S0
# --------------------------------------------------------------
TOKEN_LIMIT         = config.models.max_tokens       # max tokens per text
EMBED_CONCURRENT    = config.rate_limits.embedding_concurrent          # only 5 embedding calls in flight at once
QUESTION_CONCURRENT = config.rate_limits.question_concurrent          # only 3 question‐blocks in flight at once
_embed_semaphore    = asyncio.Semaphore(EMBED_CONCURRENT)
_question_semaphore = asyncio.Semaphore(QUESTION_CONCURRENT)


def truncate_text(
    text: str,
    max_tokens: int = TOKEN_LIMIT,
    model: str = config.models.embedding_model
) -> str:
    """
    Truncate `text` to at most TOKEN_LIMIT tokens using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens   = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text


async def get_embedding(
    client: AsyncAzureOpenAI,
    text: str,
    model: str = config.models.embedding_model
) -> List[float]:
    """
    Acquire embedding‐semaphore, then fetch one embedding for `text` from Azure.
    """
    truncated = truncate_text(text, max_tokens=TOKEN_LIMIT, model=model)
    async with _embed_semaphore:
        resp = await client.embeddings.create(
            model = model,
            input = truncated
        )
        return resp.data[0].embedding


# --------------------------------------------------------------
# 2. Pydantic model for combined pages response
# --------------------------------------------------------------
class CombinedPagesEntry(BaseModel):
    question_number: str
    pages: List[int] = Field(default_factory=list)


# --------------------------------------------------------------
# 3. Precompute page splits & embeddings per submission
# --------------------------------------------------------------
async def preprocess_submissions(
    submissions_df: pd.DataFrame,
    client: AsyncAzureOpenAI,
    embedding_model: str = config.models.embedding_model,
    backup_dir: Optional[str] = None
) -> List[Dict]:
    """
    1) If a backup CSV exists in backup_dir ("page_embeddings.csv"), load it and reconstruct results.
    2) Otherwise, flatten each submission's Markdown into pages (one row per page),
       run get_embedding(...) for each page (limited by semaphore), track per‐page progress,
       then assemble results and write a backup CSV in backup_dir.
    3) Return a list of dicts (one per submission):
         {
           "submission_id": ...,
           "original_file_name": ...,
           "pages": [...],               # List[str]
           "page_embeddings": np.ndarray # shape (num_pages, dim)
         }
    """
    backup_path = None
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, "page_embeddings.csv")

    # If backup exists, load and reconstruct
    if backup_path and os.path.isfile(backup_path):
        df_backup = pd.read_csv(backup_path)
        # Sort and group by submission
        df_backup.sort_values(
            ["submission_id", "original_file_name", "page_idx"],
            inplace=True
        )
        results: List[Dict] = []
        grouped = df_backup.groupby(["submission_id", "original_file_name"], sort=False)
        for (sid, fname), group in grouped:
            page_texts = group["page_text"].tolist()
            embed_cols = [c for c in group.columns if c.startswith("d_")]
            page_embs = group[embed_cols].to_numpy()
            results.append({
                "submission_id": sid,
                "original_file_name": fname,
                "pages": page_texts,
                "page_embeddings": page_embs
            })
        return results

    # Otherwise, compute from scratch
    # Step A: Flatten submissions into individual pages
    pages_records: List[Dict] = []
    for row in submissions_df.itertuples(index=False):
        sid, fname, full_md = row.submission_id, row.original_file_name, row.markdown
        page_texts = [p.strip() for p in full_md.split("PageBreak")]
        for idx, pg in enumerate(page_texts, start=1):
            pages_records.append({
                "submission_id": sid,
                "original_file_name": fname,
                "page_idx": idx,
                "page_text": pg
            })

    pages_df = pd.DataFrame(pages_records)
    all_texts = pages_df["page_text"].tolist()

    # Step B: Create one semaphore‐limited embedding task per page
    embed_tasks = [
        get_embedding(client, text, model=embedding_model)
        for text in all_texts
    ]

    # Step C: Gather embeddings with per-page tqdm progress
    embeddings_list: List[List[float]] = await tqdm_asyncio.gather(
        *embed_tasks,
        desc  = "Embedding pages (semaphore=5)",
        total = len(embed_tasks),
        unit  = "page"
    )

    pages_df["embedding"] = embeddings_list  # list of lists

    # Step D: If backup_dir provided, write a CSV
    if backup_path:
        dim = len(embeddings_list[0]) if embeddings_list else 0
        col_names = [f"d_{i}" for i in range(dim)]
        embed_matrix = np.array(embeddings_list)  # shape: (total_pages, dim)

        df_export = pd.DataFrame({
            "submission_id": pages_df["submission_id"],
            "original_file_name": pages_df["original_file_name"],
            "page_idx": pages_df["page_idx"],
            "page_text": pages_df["page_text"]
        })
        # Create embeddings DataFrame in one go
        embedding_df = pd.DataFrame(embed_matrix, columns=col_names)

        # Concatenate horizontally
        df_export = pd.concat([df_export, embedding_df], axis=1)

        df_export.to_csv(backup_path, index=False)

    # Step E: Reassemble per-submission page_embeddings arrays
    results: List[Dict] = []
    for row in submissions_df.itertuples(index=False):
        sid, fname, full_md = row.submission_id, row.original_file_name, row.markdown
        page_texts = [p.strip() for p in full_md.split("PageBreak")]

        sub_df = pages_df[pages_df["submission_id"] == sid].sort_values("page_idx")
        page_embs_np = np.vstack(sub_df["embedding"].tolist())

        results.append({
            "submission_id": sid,
            "original_file_name": fname,
            "pages": page_texts,
            "page_embeddings": page_embs_np
        })

    return results


# --------------------------------------------------------------
# 4. Single-question processing (LLM + embed Q/A/C)
# --------------------------------------------------------------
async def process_question_task(
    submission_id: str,
    file_name: str,
    qn: str,
    question_text: str,
    answer_text: str,
    question_context: str,
    pages: List[str],
    page_embeddings_np: np.ndarray,
    client: AsyncAzureOpenAI,
    embedding_model: str,
    model: str,
    top_k: int,
    system_prompt: str,
    token_tracker=None
) -> Dict:
    """
    1) Acquire embed‐semaphore‐limited embeddings of (question, answer, context) concurrently.
    2) Compute cosine similarities, pick top_k pages, union them.
    3) Call LLM once to format JSON response.
    """
    async with _question_semaphore:
        qac_embeddings = await asyncio.gather(
            get_embedding(client, question_text,    model=embedding_model),
            get_embedding(client, answer_text,      model=embedding_model),
            get_embedding(client, question_context, model=embedding_model)
        )
        emb_qtext   = np.array(qac_embeddings[0]).reshape(1, -1)
        emb_answer  = np.array(qac_embeddings[1]).reshape(1, -1)
        emb_context = np.array(qac_embeddings[2]).reshape(1, -1)

        sims_qtext       = cosine_similarity(emb_qtext, page_embeddings_np)[0]
        topk_qtext_idxs  = sims_qtext.argsort()[-top_k:][::-1]
        candidate_qtext_pages = [i + 1 for i in topk_qtext_idxs]

        sims_answer      = cosine_similarity(emb_answer, page_embeddings_np)[0]
        topk_ans_idxs    = sims_answer.argsort()[-top_k:][::-1]
        candidate_ans_pages = [i + 1 for i in topk_ans_idxs]

        sims_context     = cosine_similarity(emb_context, page_embeddings_np)[0]
        topk_ctx_idxs    = sims_context.argsort()[-top_k:][::-1]
        candidate_ctx_pages = [i + 1 for i in topk_ctx_idxs]

        all_candidate_pages = sorted(
            set(candidate_qtext_pages + candidate_ans_pages + candidate_ctx_pages)
        )

        combined_candidates_text = ""
        for page_num in all_candidate_pages:
            pg_text = pages[page_num - 1].replace('"', '\\"')
            combined_candidates_text += f'Page {page_num}:\n"{pg_text}"\n\n'

        user_prompt = f"""
Below are the top {len(all_candidate_pages)} candidate pages (by embedding similarity)
for Question {qn}. Each page is labeled with its page number and its entire text content.

{combined_candidates_text}

Question Number: {qn}

1) Question Text:
"{question_text.replace('"', '\\"')}"

2) Question Context (exact text needed to solve the problem):
"{question_context.replace('"', '\\"')}"

3) Student Answer (exact text):
"{answer_text.replace('"', '\\"')}"

From among these pages, identify which page numbers are required to fully capture:
  - the entire question text,
  - the entire question context,
  - and the entire student answer.

Return JSON with:
{{
  "question_number": "{qn}",
  "pages": [<list of page ints>]
}}
"""

        # --- Token tracking ---
        if token_tracker:
            token_tracker.add("map_questions_to_pages_llm", system_prompt+user_prompt)
        # ----------------------

        try:
            resp = await client.beta.chat.completions.parse(
                model = model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                response_format = CombinedPagesEntry
            )
            raw_json = resp.choices[0].message.content
            parsed   = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            chosen_pages = parsed.get("pages", [])
        except Exception:
            chosen_pages = []

        return {
            "submission_id": submission_id,
            "original_file_name": file_name,
            "question_number": qn,
            "pages": json.dumps(chosen_pages),
        }


# --------------------------------------------------------------
# 5. Main function with question‐level tqdm tracking
# --------------------------------------------------------------
async def map_questions_to_pages_llm(
    submissions_df: pd.DataFrame,
    student_answers_df: pd.DataFrame,
    client: AsyncAzureOpenAI,
    embedding_model: str = config.models.embedding_model,
    model: str = config.models.default_model,
    top_k: int = config.processing.top_k_pages,
    output_csv: str = "output.csv",
    encoder_name: str = config.models.encoder_model,
    backup_dir: Optional[str] = None,
    token_tracker=None
) -> pd.DataFrame:
    """
    1) Precompute page splits & embeddings per submission (stage 1),
       using semaphore-limited per-page embeddings and per-page tqdm.
       If backup_dir contains a CSV, load from it instead of re-embedding.
    2) Create one async task per question (batching question+answer+context),
       limited by question‐semaphore.
    3) Use tqdm_asyncio.as_completed for per-question progress.
    """
    # Stage 1: split + embed pages (with optional backup)
    submission_data = await preprocess_submissions(
        submissions_df, client, embedding_model, backup_dir
    )

    # Initialize encoder and system prompt tokens
    system_prompt = (
        "You are a strict JSON formatter. Only output valid JSON.\n"
        "Return exactly one JSON object with keys:\n"
        "  question_number (string),\n"
        "  pages (array of ints)\n"
        "No commentary, no extra keys."
    )

    # Stage 2: build a list of question-level tasks wrapped in question‐semaphore
    question_tasks = []
    for sub_info in submission_data:
        sid               = sub_info["submission_id"]
        fname             = sub_info["original_file_name"]
        pages             = sub_info["pages"]
        page_embeddings_np= sub_info["page_embeddings"]

        sa_subset = student_answers_df[
            (student_answers_df["submission_id"] == sid) &
            (student_answers_df["original_file_name"] == fname)
        ]
        for _, sa_row in sa_subset.iterrows():
            qn              = sa_row["question_number"]
            question_text   = str(sa_row["question_text"]).strip()
            answer_text     = str(sa_row["answer_text"]).strip()
            question_context= str(sa_row["question_context"]).strip()

            async def sem_task(
                sid=sid,
                fname=fname,
                qn=qn,
                question_text=question_text,
                answer_text=answer_text,
                question_context=question_context,
                pages=pages,
                page_embeddings_np=page_embeddings_np
            ):
                return await process_question_task(
                    submission_id     = sid,
                    file_name         = fname,
                    qn                = qn,
                    question_text     = question_text,
                    answer_text       = answer_text,
                    question_context  = question_context,
                    pages             = pages,
                    page_embeddings_np= page_embeddings_np,
                    client            = client,
                    embedding_model   = embedding_model,
                    model             = model,
                    top_k             = top_k,
                    system_prompt     = system_prompt,
                    token_tracker     = token_tracker
                )
                

            question_tasks.append(sem_task())

    # Stage 3: run all question tasks with tqdm_asyncio progress bar
    results: List[Dict] = []
    for coro in tqdm_asyncio.as_completed(
        question_tasks,
        total = len(question_tasks),
        desc  = "Processing each question"
    ):
        record = await coro
        results.append(record)

    df_out = pd.DataFrame.from_records(results)
    df_out.to_csv(output_csv, index=False)

    if token_tracker:
        token_tracker.print_process("map_questions_to_pages_llm")
    return df_out
