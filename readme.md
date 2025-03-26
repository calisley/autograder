# Automated Grader

This project provides an automated grading tool for student submissions. It processes student submissions, extracts questions and answers, generates an answer key and rubric, and (optionally) grades the submissions using various document processing and language model (LLM) utilities.

> **Note:** Some parts of the grading process (such as the final grading step) are currently commented out in the main code. Adjust and uncomment as needed.

---

## Features

- **Submission Conversion:** Converts PDF/DOCX submissions to Markdown.
- **Question Extraction:** Extracts questions and context from a provided assignment or from student submissions.
- **Answer Key Generation:**  
  - Converts a provided answer key document to a standardized question-level format.  
  - Alternatively, auto-generates an answer key using multiple attempts with an LLM.
- **Rubric Generation:**  
  - Processes a provided rubric document, or  
  - Auto-generates a synthetic rubric based on submissions and answer key.
- **Grading Engine:** *(Commented out in the current code)*  
  - Grades student responses against the answer key and rubric.
- **Temporary File Management:** Uses a backup folder to store intermediate files to avoid redundant processing.

---

## Dependencies

- **Python 3.7+**
- **Libraries:**
  - `os`
  - `sys`
  - `argparse`
  - `asyncio`
  - `pandas`
  - `python-dotenv` (for environment variable management)
  - `openai` (for AsyncAzureOpenAI client)
- **Custom Modules:**  
  - `llm_grader` (for grading questions)
  - `process_documents` (for processing documents)
  - `extract_problems` (for extracting questions from submissions)
  - `generate_rubric` (for creating grading rubrics)
  - `create_answer_key` (for processing answer keys)
  - `generate_answer_key` (for generating answer keys)

---

## Setup

1. **Clone the repository** and navigate to the project directory:

   ```bash
   git clone https://github.com/your-repo/automated-grader.git
   cd automated-grader
   ```
2. **Install the required packages** (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

# Set Up Environment Variables

Create a `.env` file in the root of your project with the following variables (replace with your actual values):

```
AZURE_ENDPOINT_GPT=your_azure_openai_endpoint
AZURE_API_KEY_GPT=your_azure_api_key
```

## Ensure Custom Modules are Available

The custom modules (`llm_grader`, `process_documents`, `extract_problems`, `generate_rubric`, `create_answer_key`, `generate_answer_key`) should be present in the project directory.

---

# Usage

Run the script from the command line using:

```sh
./grader.py submissions_folder [--answer_key ANSWER_KEY] [--blank_assignment BLANK_ASSIGNMENT]
            [--output_csv OUTPUT_CSV] [--rubric RUBRIC] [--truncate PAGES [PAGES ...]]
            [--backup_folder BACKUP_FOLDER]
```

## Command-Line Arguments

### Required Argument:

- **`submissions_folder`**  
  The folder containing student submissions in PDF/DOCX format.

### Optional Arguments:

- **`--answer_key`**  
  Path to the answer key document (PDF, DOCX, etc.). If not provided, an answer key will be generated automatically.

- **`--blank_assignment`**  
  Path to an unaltered copy of the assignment. If not provided, a blank assignment will be generated from a submission.

- **`--output_csv`** (Default: `./grader_output.csv`)  
  Path/filename for the CSV file that will contain the final grading results.

- **`--rubric`** (Optional)  
  Path to a rubric document. If not provided, a synthetic rubric is generated.

- **`--truncate`** (Optional)  
  List of page numbers to exclude from PDF submissions (e.g., metadata pages).

- **`--backup_folder`** (Default: `temp`)  
  Directory to store temporary files generated during processing.

---

# Workflow Overview

### Processing Submissions:
- Converts all student submissions to Markdown and stores them in `submissions_markdown.csv`.
- Creates a backup of the original submissions.

### Extracting Assignment Questions:
- Generates or retrieves a blank assignment.
- Extracts questions with context.

### Formatting Submissions:
- Converts submissions into a question-level format.

### Generating/Processing the Answer Key:
- If an answer key is provided, it is processed and standardized.
- Otherwise, an answer key is generated using multiple LLM attempts.

### Creating the Rubric:
- A provided rubric document is processed.
- If no rubric is provided, a synthetic rubric is generated.

### Grading *(Commented Out in Code)*:
- Uses the rubric and answer key to grade submissions.
- Saves the results to a CSV file.

---

# Future Enhancements

### Model Parameterization:
- Allow specifying different model versions for various functions.

### Automated Grading Execution:
- Uncomment the grading process once fully tested.

---

# Troubleshooting

### Output CSV Already Exists:
- If `grader_output.csv` exists, delete or rename it before running the script.

### Missing Files:
- Ensure required files (e.g., Markdown conversions, extracted questions) exist in the backup folder.

### Rubric and Answer Key Issues:
- Verify the format and paths of provided documents.

---
