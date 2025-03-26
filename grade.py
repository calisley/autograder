#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI


from llm_grader import grade_questions
from process_documents import process_all_documents, process_single_document
from extract_problems import process_submissions, get_questions_with_context, strip_assignment
from generate_rubric import generate_rubrics
from create_answer_key import question_level_answer_key
from generate_answer_key import generate_key, select_best_responses
from compile_feedback import generate_overall_feedback

#TODO: add the ability to pass in the model to the functions
#TODO:For when the class starts:
	# Grade suggestions 
	# Feedback about the whole quality of the assignment
	# Feedback more than what a CA would write
	# bring to “feedback for question 1”
	# AI “augmented” grading?
	# 	suggested points
	# Time saving 


async def main():
    parser = argparse.ArgumentParser(
        description="Grade student submissions against a given answer key and optional rubric."
    )

    parser.add_argument(
        "submissions_folder",
        type=str,
        help="Path to the folder containing student PDF/DOCX submissions to be converted to Markdown."
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        help="Path to the answer key document (PDF/DOCX/etc.)"
    )
    parser.add_argument(
        "--blank_assignment",
        type=str,
        help="Path to an unaltered copy of the assignment"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./grader_output.csv",
        help="Name of the CSV file to output grading results."
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Optional path to the rubric document (PDF/DOCX/etc.)."
    )
    parser.add_argument(
        "--truncate",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of page numbers to exclude from PDF submissions."
    )
    parser.add_argument(
        "--backup_folder",
        type=str,
        default="temp",
        help="Optional directory for storing temporary files from grading."
    )
    args = parser.parse_args()

    #TEMP FILES
    backup_folder = args.backup_folder

    # Ensure the temp folder exists
    os.makedirs(backup_folder, exist_ok=True)

    #stored so we don't rerun any queries uncessesarily
    submissions_csv_path = os.path.join(backup_folder, "submissions_markdown.csv")
    submissions_backup_path = os.path.join(backup_folder, "submissions_backup")
    questions_csv_path = os.path.join(backup_folder, "questions_with_answers.csv")
    blank_assignment_md_path = os.path.join(backup_folder,"blank_assignment.md")
    questions_with_context_path = os.path.join(backup_folder,"questions_with_context.csv")

    answer_key_md_path = os.path.join(backup_folder, "answer_key_backup.md")
    answer_key_csv_path = os.path.join(backup_folder, "standardized_answer_key.csv")
    answer_key_generated_answers_all_csv_path = os.path.join(backup_folder, "standardized_answer_key_all_attempts.csv")
    rubric_csv_path = os.path.join(backup_folder, "question_rubrics.csv")
    rubric_md_path = os.path.join(backup_folder, "question_rubrics.md")

    questions_markdown_path = os.path.join(backup_folder, "questions_markdown.csv")
    
    #0 Inititalize end and LLM Client:
    load_dotenv()
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT"),
        api_key=os.getenv("AZURE_API_KEY_GPT"),
        api_version="2024-12-01-preview"
    )
    #TODO: pass client to all functions
    
    if os.path.exists(args.output_csv):
        sys.exit("Output CSV already exists. Please delete or rename it before running the grader.")
    
    # 1. Process submissions
    
    # 1a. Convert all submissions in the directory to Markdown
    if not os.path.exists(submissions_csv_path):
        print("Converting submissions to Markdown...")
        submissions = await process_all_documents(
            args.submissions_folder,
            submissions_csv_path,
            submissions_backup_path,
            truncate=args.truncate,
        )
        if submissions.empty:
            print(f"No valid documents found in {args.submissions_folder}")
            sys.exit(1)
    else:
        print("Using existing submissions_markdown.csv...")
        submissions = pd.read_csv(submissions_csv_path)
    
    #1a. Get the assignment's questions. 
    if not os.path.exists(blank_assignment_md_path):
        if not args.blank_assignment:
            print("Generating blank assignment from a submission...")
            raw_assignment = await strip_assignment(submissions, output_md=blank_assignment_md_path)    
            
        else:
            #TODO: allow for the uploading of the raw assigment
            raw_assignment = ""
    else :
        print(f"Using existing blank assignment markdown from {blank_assignment_md_path}...")
        with open(blank_assignment_md_path, "r", encoding="utf-8") as f:
             raw_assignment = f.read()     
    
    #1b. Process the assignment to get questions and context as a dataframe
    if not os.path.exists(questions_with_context_path):
        questions = await get_questions_with_context(raw_assignment, client, model="o3-mini", output_csv=questions_with_context_path)
    else:
        print("Using preprocessed questions with context...")
        questions = pd.read_csv(questions_with_context_path)

    #1c. Convert submissions to question level dataframe
    if not os.path.exists(questions_csv_path):
        print("Formatting submissions...")
        submission_by_question = await process_submissions(submissions,questions,client,questions_csv_path, "o3-mini")
    else:
        submission_by_question = pd.read_csv(questions_csv_path)

    #3. Generate the answer key
    if not os.path.exists(answer_key_csv_path):
        if args.answer_key:
        #3a. If provided, convert answer key to markdown
            if not os.path.exists(answer_key_md_path):
                print("Extracting answer key from provided document...")
                answer_key_md = await process_single_document(
                    args.answer_key,
                    output_md_path=answer_key_md_path
                )
            else:
                print(f"Using existing answer key markdown from {answer_key_md_path}...")
                with open(answer_key_md_path, "r", encoding="utf-8") as f:
                    answer_key_md = f.read()         
            #Then, convert it to question level
            answer_key_df = await question_level_answer_key(
                    answer_key_md, 
                    output_csv=answer_key_csv_path
            )  
     #3b If not provided, generate an answer key
        else:
            if not os.path.exists(answer_key_generated_answers_all_csv_path):
                generated_answers = await generate_key(
                    questions,
                    n_attempts=5, 
                    model="o3-mini", 
                    output_csv=answer_key_generated_answers_all_csv_path
                    )
            else:
                generated_answers = pd.read_csv(answer_key_generated_answers_all_csv_path)

            answer_key_df = await select_best_responses(
                generated_answers,
                client,
                model="gpt-4o",
                output_csv=answer_key_csv_path
            )
      
    else:
        print(f"Using existing answer key CSV from {answer_key_csv_path}...")
        answer_key_df = pd.read_csv(answer_key_csv_path)
    
    # 4. Create the rubric
    if not os.path.exists(rubric_csv_path):
        if args.rubric:
            print("Fetching rubric...")
            if not os.path.isfile(args.rubric):
                print(f"Rubric file {args.rubric} cannot be found.")
                sys.exit(1)
            else:
                rubric_md = await process_single_document(args.rubric)
                #TODO:convert rubric to question level
                if not rubric_md:
                    print(f"Failed to process rubric document: {args.rubric}")
                    rubric_md = ""
        else:
            # No rubric provided, or auto-generate a synthetic rubric
            print("Generating rubric...")
            rubric_df = await generate_rubrics(submission_by_question, answer_key_df, client, sample_size=10, model="gpt-4o", output_csv=rubric_csv_path, output_md=rubric_md_path)
    else:
        print(f"Using existing rubric CSV from {rubric_csv_path}...")
        rubric_df = pd.read_csv(rubric_csv_path)


    # 5. Perform the grading
    print("Grading assignments...")

    results_df = await grade_questions(
        submission_by_question,
        answer_key_df,
        rubric_df,
        client,
        model="o3-mini",
        batch_size=20
    )

    if results_df.empty:
        print("No grading results were returned. Check your grader logic.")
        sys.exit(1)

    # 6. Save results to the specified CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Grading complete! Results saved to {args.output_csv}")

    #results_df = pd.read_csv("/Users/cai529/Github/autograder/trials/dpi-assignment-1/assignment_1_grades_1.csv")
    # 7. Generate higher level feedback
    print("Generating 'AI TF' feedback...")
    overall_feedback = await generate_overall_feedback(
        results_df,
        client,
        model="o3-mini",
        batch_size=10
    )
    overall_feedback.to_csv("./feeedback.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
