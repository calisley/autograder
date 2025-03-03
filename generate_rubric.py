
import asyncio
import pandas as pd
from tqdm import tqdm
import random
from openai import AsyncAzureOpenAI

async def generate_rubric_for_question(question_text: str, question_context:str, question_answer:str, question_explanation:str, points: int, sample_answers: list, 
                                      openai_client: AsyncAzureOpenAI, model: str = "gpt-4o") -> str:
    """
    Generate a detailed rubric for a question based on sample student answers.
    
    Args:
    
       
        question_text: The text of the question
        question_context=Additional necessary context to answer the question,
        question_answer=The correct answer,
        question_explanation=Potential explanation,
        points: Maximum points for the question
        sample_answers: List of sample student answers
        openai_client: Azure OpenAI client
        model: Model to use for generation
        
    Returns:
        Detailed rubric as a string
    """
    system_prompt = (
        "You are an expert educator and grader. Your task is to create a detailed, fair, comprehensive yet concise grading rubric based on a question and sample student answers."
        "IMPORTANT: Student responses have been collected using OCR. When considering student responses, assume typos, spelling errors, formatting issues, or contradictory answers are attributable to OCR errors, not the student. OCR struggles to detect slashthroughs, so a student crossing out an answer may not be reflected in the text."
        "Errors potentially related to OCR should not be part of the rubruic. For example, do not penalize a student for failing to underline something, as they could have, and it was not detected."
        "Do your best to interpret the student's intent, and use that in the creation of your rubric"
    )
    
    # Format sample answers for the prompt
    formatted_answers = "\n\n".join([f"SAMPLE ANSWER {i+1}:\n{answer}" for i, answer in enumerate(sample_answers)])
    
    user_prompt = f"""
    QUESTION CONTEXT: {question_context}\n

    QUESTION (worth {points} points):
    {question_text}\n
    
    SAMPLE STUDENT ANSWERS:
    {formatted_answers}
    
    SUGGESTED ANSWER: {question_answer}\n
    SUGGESTED ANSWER EXPLANATION: {question_explanation}\n

    Based on the question, these sample answers, and the correct answer/explanation create a detailed but concise rubric that:
    1. Breaks down how the {points} points should be allocated
    2. Specifies what earns full credit (+points)
    3. Provides clear criteria for partial credit
    Your rubric should always only contain these 3 sections. 
    
    Important Notes:
    - If questions allow for open ended explanation, allow for convicing arguments of an incorrect answer to be awarded full points, unless the answer is factually incorrect. Suggested do not always represent the only correct answer.
    - If a grading question requires resources that you are incapable of (e.g. following a link, viewing an image, etc.), 
        create a rubric that assigns points for what you can see (e.g, they included a link in their response, you have some text from the image, etc.) 
        In that rubric, you MUST include an indication that the question cannot be fully graded by an AI, and requires human evaluation. 
        When allocating points for these questions, denote which parts can be graded by an AI, and which require human attention.
    - IMPORTANT: Only mention human evaluation if there are parts of the answer you cannot grade on a functionality basis (again, that is following a link, viewing an image, etc.). Interpreting student responses is still part of the AI's job.

    - Do not create requirements that are not present in the question or question context. For example, if a question does not ask for a specific number of examples, the rubric should not aware more (or less) points for a specific number of examples.
    - Unless explicitly mentioned in the question (note, plurals as an indication of quantity are not explicit) do not penalize or give credit for quantity. Only focus on quality.     
    Format your rubric with clear point allocations (e.g., "+2 points for...", "-1 point for..."). 

    Do not grade the answers themselves, only create a rubric. Do not return the question text in the rubric, only the grading criteria. Do not provide any other commentary. 
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating rubric: {e}")
        return f"Failed to generate rubric: {str(e)}"

async def generate_rubrics(questions_and_answers: pd.DataFrame, answer_key: pd.DataFrame, client: AsyncAzureOpenAI,
                        output_csv: str = "./temp/question_rubrics.csv", output_md: str = "./temp/question_rubrics.md",
                          sample_size: int = 10, model: str = "gpt-4o") -> pd.DataFrame:
    """
    Generate rubrics for each unique question in the dataset.
    
    Args:
        questions_and_answers: Dataframe of the student's answers at the question level
        anser_key: Dataframe of the answer key at the question level
        output_csv: Path to save generated rubrics as JSON
        output_md: Path to save generated rubrics as Markdown
        sample_size: Number of sample answers to consider per question
        model: Model to use for generation
        
    Returns:
        DataFrame with columns: question_number, points, and rubric
    """
    questions_and_answers = questions_and_answers[['question_number', 'submission_id', 'answer_text']]
    # Load questions dataframe
    df = answer_key.merge(questions_and_answers, on="question_number", how="left")
    # Get unique questions
    unique_questions = df.drop_duplicates(subset=['question_number'])
    
    rubrics = {}
    
    async def process_question(row):
        question_number = row['question_number']
        question_text = row['question_text']
        question_context = row['question_context']
        question_answer = row['best_answer']
        question_explanation = row['best_explanation']
        points = row['points']
        
        # Get sample answers for this question (up to sample_size)
        sample_answers = df[df['question_number'] == question_number]['answer_text'].tolist()
        if len(sample_answers) > sample_size:
            sample_answers = sample_answers[:sample_size]
        
        # Generate rubric
        rubric = await generate_rubric_for_question(
            question_text=question_text,
            question_context=question_context,
            question_answer=question_answer,
            question_explanation=question_explanation,
            points=points,
            sample_answers=sample_answers,
            openai_client=client,
            model=model
        )
        
        rubrics[question_number] = {
            'question_text': question_text,
            'question_context': question_context,
            'question_answer': question_answer,
            'question_explanation': question_explanation,
            'points': points,
            'rubric': rubric
        }
    
    # Create tasks for each unique question
    tasks = [process_question(row) for _, row in unique_questions.iterrows()]
    
    # Run tasks with progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating rubrics"):
        await coro
    
    print("First rubric created. Now validating rubrics...")
    rubrics = await validate_rubrics(rubrics, questions_and_answers, openai_client=client, model=model, batch_size=10)
    print("Rubrics validated.")
    # Create a DataFrame with the final (validated) rubrics.
    rubrics_df = pd.DataFrame([
        {
            'question_number': q_num,
            'points': data['points'],
            'rubric': data['rubric']
        }
        for q_num, data in rubrics.items()
    ])

    rubrics_df.to_csv(output_csv, index=False)
    
    with open(output_md, 'w', encoding='utf-8') as md_file:
        md_file.write("# Question Rubrics\n\n")
        for q_num, data in sorted(rubrics.items()):
            md_file.write(f"## Question {q_num} ({data['points']} points)\n\n")
            md_file.write(f"### Rubric\n\n{data['rubric']}\n\n")
            md_file.write("---\n\n")
    
    print(f"Rubrics saved to {output_csv} and {output_md}")
    return rubrics_df




async def validate_rubric_for_batch(question_text: str, question_context: str, question_answer: str, 
                                    question_explanation: str, points: int, student_answers: list, 
                                    current_rubric: str, openai_client: AsyncAzureOpenAI, model: str = "gpt-4o") -> str:
    """
    Evaluate a batch of student answers against the current rubric.
    If the rubric is inadequate for any answer in the batch, return an updated rubric that incorporates
    these student answers while preserving the existing criteria.
    
    Args:
        question_text: The text of the question.
        question_context: Additional context for the question.
        question_answer: The correct answer.
        question_explanation: Explanation for the correct answer.
        points: Total points for the question.
        student_answers: A list of up to 10 student answers.
        current_rubric: The current rubric.
        openai_client: Azure OpenAI client.
        model: The model to use for generation.
        
    Returns:
        The current rubric if adequate; otherwise, an updated rubric.
    """
    system_prompt = (
        "You are an expert educator and grader. Evaluate the following rubric for grading student answers."
    )
    
    # Format the batch of student answers
    formatted_answers = "\n\n".join([f"STUDENT ANSWER {i+1}:\n{ans}" for i, ans in enumerate(student_answers)])
    
    user_prompt = f"""
    QUESTION CONTEXT: {question_context}
    
    QUESTION (worth {points} points):
    {question_text}
    
    SUGGESTED ANSWER: {question_answer}
    
    SUGGESTED ANSWER EXPLANATION: {question_explanation}
    
    CURRENT RUBRIC:
    {current_rubric}
    
    STUDENT ANSWERS:
    {formatted_answers}
    
    Evaluate whether the current rubric adequately covers the criteria needed to grade all of the above student answers. 
    If the rubric is adequate, simply respond with the word "adequate".

    If it is not adequate, provide an updated rubric that incorporates these student answers while preserving the previously established criteria.
    The updated rubric should follow the same structure:
        1. Breaks down how the points should be allocated
        2. Specifies what earns full credit (+points)
        3. Provides clear criteria for partial credit
    
    Important Notes:
    - If the rubric mentions requiring human evaluation, that is not an indication that the rubric is inadequate. 
    - If questions allow for open ended explanation, allow for convicing arguments of an incorrect answer to be awarded full points, unless the answer is factually incorrect. Suggested do not always represent the only correct answer.
    - If a grading question requires resources that you are incapable of (e.g. following a link, viewing an image, etc.), 
        create a rubric that assigns points for what you can see (e.g, they included a link in their response, you have some text from the image, etc.) 
        In that rubric, you MUST include an indication that the question cannot be fully graded by an AI, and requires human evaluation. 
        However, you should still create a rubric for criteria that the human should use in its evaluation, but clearly mark that as only for humans. 
    - Do not create requirements that are not present in the question or question context. For example, if a question does not ask for a specific number of examples, the rubric should not aware more (or less) points for a specific number of examples.
    - Only mention human evaluation if it is necessary for grading the question. 



    Do not call the rubric the "updated rubric" or include any additional commentary. Do not include sample student answers in your response.
    Do not include any additional commentary.
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during batch rubric validation: {e}")
        return current_rubric
    

async def validate_rubrics(rubrics: dict, submissions: pd.DataFrame, openai_client: AsyncAzureOpenAI,
                           model: str = "gpt-4o", batch_size: int = 5) -> dict:
    """
    Validates and updates rubrics based on student submissions in batches of 10.
    
    For each question in the rubrics dictionary:
      - Group the student submissions in batches of up to 10.
      - Ask an LLM whether the current rubric adequately grades the batch.
      - If the rubric is inadequate, update it based on the batch of student answers.
    
    Args:
        rubrics: Dictionary where each key is a question_number and each value is a dict containing question details and its rubric.
        submissions: DataFrame containing columns including 'question_number' and 'answer_text'.
        openai_client: Azure OpenAI client.
        model: The model to use for generation.
        batch_size: Number of student answers to process per batch (default is 10).
        
    Returns:
        Updated rubrics dictionary.
    """
        # Iterate over each question in the rubrics with a progress bar
    async def process_question(q_num: str, data: dict):
            # Get all submissions for the current question
            sub_list = submissions[submissions['question_number'] == q_num]['answer_text'].tolist()
            if len(sub_list) > 3 * batch_size:
                question_submissions = random.sample(sub_list, 3 * batch_size)
            else:
                question_submissions = sub_list
            
            current_rubric = data['rubric']
            #num_batches = (len(question_submissions) + batch_size - 1) // batch_size
            
            # Process student answers in batches sequentially for this question
            for i in range(0, len(question_submissions), batch_size):
                batch = question_submissions[i:i+batch_size]
                result = await validate_rubric_for_batch(
                    question_text=data['question_text'],
                    question_context=data['question_context'],
                    question_answer=data['question_answer'],
                    question_explanation=data['question_explanation'],
                    points=data['points'],
                    student_answers=batch,
                    current_rubric=current_rubric,
                    openai_client=openai_client,
                    model=model
                )
                
                # If the response is not "adequate", update the rubric for this question.
                if result.lower() != "adequate":
                    current_rubric = result
            
            # Return the final rubric for this question.
            return q_num, current_rubric

        # Launch an async task for each question.
    tasks = [process_question(q_num, data) for q_num, data in rubrics.items()]
        
    # Use tqdm to show progress over all questions.
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Validating rubrics per question"):
        q_num, updated_rubric = await future
        rubrics[q_num]['rubric'] = updated_rubric
    
    return rubrics


# Example usage:
# updated_rubrics = await validate_rubrics(rubrics, questions_and_answers, client, model="gpt-4o")
#python3 grade.py ./trials/dpi-assignment-2/Assignment\ 2\ Submissions --output_csv ./trials/dpi-assignment-2/assignment_2_grades.csv --backup_folder ./trials/dpi-assignment-2/temp_dpi/