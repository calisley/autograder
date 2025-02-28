import os
import asyncio
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

async def generate_rubric_for_question(question_text: str, points: int, sample_answers: list, 
                                      openai_client: AsyncAzureOpenAI, model: str = "gpt-4o") -> str:
    """
    Generate a detailed rubric for a question based on sample student answers.
    
    Args:
        question_text: The text of the question
        points: Maximum points for the question
        sample_answers: List of sample student answers
        openai_client: Azure OpenAI client
        model: Model to use for generation
        
    Returns:
        Detailed rubric as a string
    """
    system_prompt = (
        "You are an expert educator and grader. Your task is to create a detailed, fair, and comprehensive "
        "grading rubric based on a question and sample student answers."
    )
    
    # Format sample answers for the prompt
    formatted_answers = "\n\n".join([f"SAMPLE ANSWER {i+1}:\n{answer}" for i, answer in enumerate(sample_answers)])
    
    user_prompt = f"""
    QUESTION (worth {points} points):
    {question_text}
    
    SAMPLE STUDENT ANSWERS:
    {formatted_answers}
    
    Based on the question and these sample answers, create a detailed rubric that:
    1. Breaks down how the {points} points should be allocated
    2. Specifies what earns full credit (+points)
    3. Lists common mistakes and misconceptions (-points)
    4. Provides clear criteria for partial credit
    
    Format your rubric with clear point allocations (e.g., "+2 points for...", "-1 point for...").
    Make the rubric detailed enough that different graders would assign similar scores.

    Do not grade the answers themselves, only create a rubric.

    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating rubric: {e}")
        return f"Failed to generate rubric: {str(e)}"

async def generate_rubrics(questions_csv: str, output_csv: str = "question_rubrics.csv", 
                          output_md: str = "question_rubrics.md",
                          sample_size: int = 10, model: str = "gpt-4o") -> pd.DataFrame:
    """
    Generate rubrics for each unique question in the dataset.
    
    Args:
        questions_csv: Path to CSV with questions and answers
        output_csv: Path to save generated rubrics as JSON
        output_md: Path to save generated rubrics as Markdown
        sample_size: Number of sample answers to consider per question
        model: Model to use for generation
        
    Returns:
        DataFrame with columns: question_number, points, and rubric
    """
    load_dotenv()
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT"),
        api_key=os.getenv("AZURE_API_KEY_GPT"),
        api_version="2024-02-15-preview"
    )
    
    # Load questions dataframe
    df = pd.read_csv(questions_csv)
    
    # Get unique questions
    unique_questions = df.drop_duplicates(subset=['question_number'])
    
    rubrics = {}
    
    async def process_question(row):
        question_number = row['question_number']
        question_text = row['question_text']
        points = row['points']
        
        # Get sample answers for this question (up to sample_size)
        sample_answers = df[df['question_number'] == question_number]['answer_text'].tolist()
        if len(sample_answers) > sample_size:
            sample_answers = sample_answers[:sample_size]
        
        # Generate rubric
        rubric = await generate_rubric_for_question(
            question_text=question_text,
            points=points,
            sample_answers=sample_answers,
            openai_client=client,
            model=model
        )
        
        rubrics[question_number] = {
            'question_text': question_text,
            'points': points,
            'rubric': rubric
        }
    
    # Create tasks for each unique question
    tasks = [process_question(row) for _, row in unique_questions.iterrows()]
    
    # Run tasks with progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating rubrics"):
        await coro
    
    
    # Create a DataFrame with the rubrics
    rubrics_df = pd.DataFrame([
        {
            'question_number': q_num,
            'points': data['points'],
            'rubric': data['rubric']
        }
        for q_num, data in rubrics.items()
    ])
    rubrics_df.to_csv(output_csv, index=False)
    
    # Create and save markdown file for human readability
    with open(output_md, 'w', encoding='utf-8') as md_file:
        md_file.write("# Question Rubrics\n\n")
        for q_num, data in sorted(rubrics.items()):
            md_file.write(f"## Question {q_num} ({data['points']} points)\n\n")
            md_file.write(f"### Question Text\n\n{data['question_text']}\n\n")
            md_file.write(f"### Rubric\n\n{data['rubric']}\n\n")
            md_file.write("---\n\n")
    
    print(f"Rubrics saved to {output_csv} and {output_md}")
    return rubrics_df

# Example usage:
# rubrics_df = await generate_rubrics("./questions_markdown.csv", sample_size=10, model="gpt-4o")
