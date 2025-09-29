#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from docx2pdf import convert
import concurrent.futures

from processing.grading.llm_grader import grade_questions, grade_questions_simple
from processing.document_ingest.process_documents import process_all_documents, process_single_document
from processing.extraction.extract_problems import process_submissions, get_questions_with_context, strip_assignment
from processing.rubric_answer_key.generate_rubric import generate_rubrics, expand_rubric
from processing.rubric_answer_key.create_answer_key import question_level_answer_key
from processing.rubric_answer_key.generate_answer_key import generate_key, select_best_responses
from processing.grading.compile_feedback import generate_subquestion_feedback
from processing.extraction.replace_pp_link import replace_pingpong_urls_in_submissions
from processing.extraction.get_page_nums import map_questions_to_pages_llm
from processing.document_ingest.pdf2img import create_images
from helpers.token_tracker import token_tracker
from config import config

#TODO: Send pages to LLM
#TODO: Redo with proper structured outputs

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
    parser.add_argument(
        "--threads_file",
        type=str,
        default=None,
        help="Optional: Path to a CSV file containing PingPong threads data. If provided, will replace links with conversation text."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Azure OpenAI model to use for grading (default: gpt-5-mini)"
    )

    args = parser.parse_args()
    model = args.model

    #TEMP FILES
    # Set backup folder to input_dir's parent folder
    input_dir_parent = os.path.dirname(os.path.abspath(args.submissions_folder))
    backup_folder = os.path.join(input_dir_parent, args.backup_folder)

    # Ensure the temp folder exists
    os.makedirs(backup_folder, exist_ok=True)

    #stored so we don't rerun any queries uncessesarily
    submissions_csv_path = os.path.join(backup_folder, "submissions_markdown.csv")
    submissions_backup_path = os.path.join(backup_folder, "submissions_backup")
    questions_csv_path = os.path.join(backup_folder, "questions_with_answers.csv")
    questions_pp_subbed = os.path.join(backup_folder, "questions_with_pp_threads.csv")

    blank_assignment_md_path = os.path.join(backup_folder,"blank_assignment.md")
    questions_with_context_path = os.path.join(backup_folder,"questions_with_context.csv")

    img_dir = os.path.join(backup_folder, "img")
    os.makedirs(img_dir, exist_ok=True)    
    

    answer_key_md_path = os.path.join(backup_folder, "answer_key_backup.md")
    answer_key_csv_path = os.path.join(backup_folder, "standardized_answer_key.csv")
    answer_key_generated_answers_all_csv_path = os.path.join(backup_folder, "standardized_answer_key_all_attempts.csv")
    rubric_csv_path = os.path.join(backup_folder, "question_rubrics.csv")
    rubric_md_path = os.path.join(backup_folder, "question_rubrics.md")

    questions_markdown_path = os.path.join(backup_folder, "questions_markdown.csv")
    question_page_mapping_path = os.path.join(backup_folder, "question_page_mapping.csv")

    #0 Initialize and LLM Client:
    load_dotenv()
    
    # Validate configuration
    config.validate()
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=config.azure.endpoint_gpt,
        api_key=config.azure.api_key_gpt,
        api_version=config.azure.api_version
    )
    
    # if os.path.exists(args.output_csv):
    #     sys.exit("Output CSV already exists. Please delete or rename it before running the grader.")
    
    # 1. Process submissions
    # 1a. Validate and convert files to PDF if needed, then convert to Markdown

    if not os.path.exists(submissions_csv_path):
        # Check if submissions folder exists
        if not os.path.exists(args.submissions_folder):
            print(f"Error: Submissions folder '{args.submissions_folder}' does not exist.")
            sys.exit(1)
        
        # Validate and convert files
        print("Validating and converting files...")
        try:
            convert(args.submissions_folder)
            print("Successfully converted DOCX files to PDF")
        except Exception as e:
            print(f"Error converting DOCX files: {e}")
            print("Please ensure all files are either PDF or DOCX format")
            sys.exit(1)
        
        # Launch create_images in the background using a thread
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            print("Starting create_images...")
            image_future = loop.run_in_executor(pool, create_images, args.submissions_folder, img_dir)

            print("Converting submissions to Markdown...")
            #TODO: Figure out how to track costs of document intelligence
            submissions = await process_all_documents(
                args.submissions_folder,
                submissions_csv_path,
                submissions_backup_path,
                truncate=args.truncate,
            )
           
            if submissions.empty:
                print(f"No valid documents found in {args.submissions_folder}")
                sys.exit(1)

            await image_future  # Wait for image generation to complete
    else:
        print("Using existing submissions_markdown.csv...")
        submissions = pd.read_csv(submissions_csv_path)
      
    
    #1a. Get the assignment's questions. 
    if not os.path.exists(blank_assignment_md_path):
        print(args.blank_assignment)
        if not args.blank_assignment:
            print("Generating blank assignment from a submission...")
            raw_assignment = await strip_assignment(submissions, 
                                                    output_md=blank_assignment_md_path, 
                                                    client = client, 
                                                    token_tracker=token_tracker)    
        else:
            print("Casting blank assignment to markdown")
            raw_assignment = await process_single_document(args.blank_assignment,
                                                           blank_assignment_md_path)

    else :
        print(f"Using existing blank assignment markdown from {blank_assignment_md_path}...")
        with open(blank_assignment_md_path, "r", encoding="utf-8") as f:
             raw_assignment = f.read()     

    #1b. Process the assignment to get questions and context as a dataframe
    if not os.path.exists(questions_with_context_path):
        questions = await get_questions_with_context(raw_assignment, 
                                                     client, 
                                                     model=model, 
                                                     output_csv=questions_with_context_path, 
                                                     token_tracker=token_tracker)
    else:
        print("Using preprocessed questions with context...")
        questions = pd.read_csv(questions_with_context_path)

    #1c. Convert submissions to question level dataframe
    if not os.path.exists(questions_csv_path):
        print("Formatting submissions...")
        submission_by_question = await process_submissions(submissions, 
                                                           questions, 
                                                           client, 
                                                           questions_csv_path, 
                                                           model, 
                                                           token_tracker=token_tracker)
    else:
        submission_by_question = pd.read_csv(questions_csv_path)

    if not os.path.exists(question_page_mapping_path):
        print("Extracting page numbers from submissions...")
        with_page_numbers = await map_questions_to_pages_llm(
            submissions,
            submission_by_question, 
            client, 
            output_csv=question_page_mapping_path, 
            model=model, 
            backup_dir=backup_folder,
            token_tracker=token_tracker
        )
    else:
        print(f"Using existing question-page mapping from {question_page_mapping_path}...")
        with_page_numbers = pd.read_csv(question_page_mapping_path)

    # 1d. Swap URLS for PingPong Chats
    if args.threads_file:
        print("Replacing PingPong URLs with conversation text...")
        submission_by_question = replace_pingpong_urls_in_submissions(submission_by_question, args.threads_file)
        submission_by_question.to_csv(questions_pp_subbed, index=False)

    # #3. Generate the answer key
    # if not os.path.exists(answer_key_csv_path):
    #     if args.answer_key:
    #     #3a. If provided, convert answer key to markdown
    #         if not os.path.exists(answer_key_md_path):
    #             print("Extracting answer key from provided document...")
    #             answer_key_md = await process_single_document(
    #                 args.answer_key,
    #                 output_md_path=answer_key_md_path
    #             )
    #         else:
    #             print(f"Using existing answer key markdown from {answer_key_md_path}...")
    #             with open(answer_key_md_path, "r", encoding="utf-8") as f:
    #                 answer_key_md = f.read()         
    #         #Then, convert it to question level
    #         answer_key_df = await question_level_answer_key(
    #                 answer_key_md, 
    #                 output_csv=answer_key_csv_path
    #         )  
    #  #3b If not provided, generate an answer key
    #     else:
    #         if not os.path.exists(answer_key_generated_answers_all_csv_path):
    #             generated_answers = await generate_key(
    #                 questions,
    #                 n_attempts=5, 
    #                 model=model, 
    #                 output_csv=answer_key_generated_answers_all_csv_path
    #                 )
    #         else:
    #             generated_answers = pd.read_csv(answer_key_generated_answers_all_csv_path)

    #         answer_key_df = await select_best_responses(
    #             generated_answers,
    #             client,
    #             model="gpt-4o",
    #             output_csv=answer_key_csv_path
    #         )
      
    # else:
    #     print(f"Using existing answer key CSV from {answer_key_csv_path}...")
    #     answer_key_df = pd.read_csv(answer_key_csv_path)
    
    # 4. Create the rubric
    if not os.path.exists(rubric_csv_path):
        if args.rubric:
            print("Fetching rubric...")
            if not os.path.isfile(args.rubric):
                print(f"Rubric file {args.rubric} cannot be found.")
                sys.exit(1)
            else:
                rubric_md = await process_single_document(args.rubric)
                print("Expanding given rubric...")
                rubric_df = await expand_rubric(rubric_md, 
                                                questions, 
                                                client, 
                                                model=model, 
                                                output_csv=rubric_csv_path, 
                                                token_tracker=token_tracker)
                if not rubric_md:
                    print(f"Failed to process rubric document: {args.rubric}")
                    rubric_md = ""
        else:
            # No rubric provided, or auto-generate a synthetic rubric
            print("Generating rubric...")
            #rubric_df = await generate_rubrics(submission_by_question, answer_key_df, client, sample_size=10, model="gpt-4o", output_csv=rubric_csv_path, output_md=rubric_md_path)
    else:
        print(f"Using existing rubric CSV from {rubric_csv_path}...")
        rubric_df = pd.read_csv(rubric_csv_path)

    # 5. Perform the grading
    print("Grading assignments...")  
    # initial, full-feedback pass (keeps the long JSON etc.)
    results_df = await grade_questions(
        submission_by_question,
        questions,
        rubric_df,
        client,
        model=model,
        page_mapping=with_page_numbers,
        img_dir=img_dir,
        token_tracker=token_tracker
    )
    results_df.to_csv(args.output_csv, index=False)
    token_tracker.print_grand_total()
    # add four "grade-only" passes, each in its own column
    for i in range(1, 3):
        results_df = await grade_questions_simple(
            results_df,
            client,
            n=i,
            model=model,
            bar_desc=f"Quick grade pass {i}", 
            token_tracker=token_tracker
        )

    if results_df.empty:
        print("No grading results were returned. Check your grader logic.")
        sys.exit(1)

    # 6. Save results to the specified CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Grading complete! Results saved to {args.output_csv}")
    #results_df = pd.read_csv("/Users/cai529/Github/autograder_dpi681/trials/assignment_2/grades.csv")
    # 7. Generate higher level feedback
    print("Generating 'AI TF' feedback...")
    overall_feedback = await generate_subquestion_feedback(
        results_df,
        client,
        model=model,
        token_tracker=token_tracker
    )
    # Save feedback to input_dir's parent folder
    feedback_output_path = os.path.join(input_dir_parent, "feedback.csv")
    overall_feedback.to_csv(feedback_output_path, index=False)
    print(f"Feedback saved to {feedback_output_path}")

    # Print the grand total tokens at the end
    token_tracker.print_grand_total()


if __name__ == "__main__":
    asyncio.run(main())
