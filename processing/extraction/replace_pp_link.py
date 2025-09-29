import pandas as pd
import re

def replace_pingpong_urls_in_submissions(
    submission_by_question: pd.DataFrame,
    threads_file: str
) -> pd.DataFrame:
    """
    For every answer in `submission_by_question['answer_text']`, find any
    PingPong thread URL of the form
        https://pingpong.hks.harvard.edu/group/<group_id>/thread/<thread_id>
    and append the full conversation from `threads_file` *after* the URL,
    leaving the original URL intact.

    Parameters
    ----------
    submission_by_question : pd.DataFrame
        Must contain a column named 'answer_text'.
    threads_file : str
        CSV exported from PingPong; must have columns
        ['Thread ID', 'Role', 'Content'].

    Returns
    -------
    pd.DataFrame
        Same object with its 'answer_text' column modified in-place.
    """
    # Load threads file
    threads_df = pd.read_csv(threads_file)

    # Build a lookup: thread_id (str) -> formatted conversation string
    thread_conversations = {}
    for thread_id, group in threads_df.groupby("Thread ID"):
        conversation_lines = [
            f"[{row['Role']}]: {row['Content']}" for _, row in group.iterrows()
        ]
        thread_conversations[str(thread_id)] = "\n".join(conversation_lines)

    # Pattern that captures the thread_id so we can look it up
    pattern = re.compile(
        r"https://pingpong\.hks\.harvard\.edu/group/\d+/thread/(\d+)"
    )

    def replace_urls(text: str) -> str:
        """Append conversation after every PingPong URL found in *text*."""
        if not isinstance(text, str):
            return text  # leave NaNs or non-strings untouched

        def repl(match: re.Match) -> str:
            thread_id = match.group(1)  # captured by (\d+)
            convo = thread_conversations.get(
                thread_id, f"[Missing thread {thread_id}]"
            )
            # Keep the original link, then paste the convo below it.
            return f"{match.group(0)}\n\n{convo}\n"

        return pattern.sub(repl, text)

    # Apply to every answer
    submission_by_question["answer_text"] = (
        submission_by_question["answer_text"].apply(replace_urls)
    )

    return submission_by_question
