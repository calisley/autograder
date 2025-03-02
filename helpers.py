import pandas as pd

def seperate_assignment(df):
    """
    Returns a DataFrame containing only the question text, question number and points allowed for a homework problem. 

    Parameters:
    df (pd.DataFrame): A DataFrame with at least columns:
                       'submission_id', 'question_text', 'question_number', 
                       'answer_text', and 'points'.

    Returns:
    pd.DataFrame: Filtered DataFrame with only the first submission_id's data 
                  and without the 'answer_text' column.
    """
    if "submission_id" not in df.columns:
        raise ValueError("The DataFrame must contain a 'submission_id' column.")

    # Get the first submission_id
    first_submission_id = df["submission_id"].iloc[0]

    # Filter data for the first submission_id
    filtered_df = df[df["submission_id"] == first_submission_id].drop(columns=["answer_text"])

    return filtered_df
