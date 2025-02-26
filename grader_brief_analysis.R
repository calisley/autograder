library(dplyr)
library(tidyr)
library(tidyverse)

# Load data
df <- read.csv("/Users/cai529/Github/autograder/question_level_grades.csv")
real <- read.csv("/Users/cai529/Github/autograder/API-201-AB-_Final-Stage_1_scores.csv")

# Rename real grades columns
colnames(real) <- gsub("^X(\\d+)..\\d+..[0-9\\.]+\\.pts\\.$", "q_\\1", colnames(real))

# Compute total LLM scores
df_total_score <- df %>%
  group_by(submission_id) %>%
  summarize(total_score_llm = sum(points_awarded),
            max_points_llm = sum(total_points))

# Reshape question-level points to wide format
df_points <- df %>%
  pivot_wider(
    id_cols = submission_id,
    names_from = question_num,
    values_from = points_awarded,
    names_prefix = "q_"
  )

# Merge LLM scores with reshaped question scores
df_wide <- df_total_score %>%
  left_join(df_points, by = "submission_id")

# Merge with real scores by submission_id
df_final <- df_wide %>%
  left_join(real, by = c("submission_id"="Submission.ID"), suffix = c("_llm","_human"))

# Compute differences
df_final <- df_final %>%
  mutate(total_score_diff = total_score_llm - Total.Score)  # Difference in total score
# Identify question columns in LLM and human datasets
# Extract question numbers from LLM columns (assuming they start with "q_")
question_numbers <- gsub("^q_(\\d+).*", "\\1", grep("^q_\\d+", colnames(df_final), value = TRUE))
question_numbers <- unique(question_numbers)  # Ensure unique question numbers

# Compute per-question differences: LLM - Human
for (q in question_numbers) {
  df_final[[paste0("q_", q, "_diff")]] <- df_final[[paste0("q_", q, "_llm")]] - df_final[[paste0("q_", q, "_human")]]
}

# Compute summary statistics for total score
summary_stats <- df_final %>%
  summarize(
    avg_total_score_diff = mean(total_score_diff, na.rm = TRUE),  # Average difference in total score
    var_total_score_diff = var(total_score_diff, na.rm = TRUE)    # Variance of total score differences
  )

# Compute per-question statistics (mean & variance of differences)
question_diffs <- df_final %>%
  summarize(across(ends_with("_diff"), list(
    avg = ~ mean(.x, na.rm = TRUE),
    var = ~ var(.x, na.rm = TRUE)
  ), .names = "{.col}_{.fn}"))

question_diffs_long <- question_diffs %>%
  pivot_longer(
    cols = everything(),
    names_to = c("question", ".value"),
    names_pattern = "q_(\\d+)_diff_(.*)"
  ) %>%
  mutate(question = as.integer(question)) %>%  # Convert question numbers to integer
  rename(mean_diff = avg, variance_diff = var) %>%
  arrange(question)  # Sort by question number

library(broom)  # For organizing test results

# 1. T-test for total score difference
t_test_total <- t.test(df_final$total_score_diff, mu = 0, na.action = na.omit)

# 2. T-tests for each question
t_test_questions <- map_dfr(question_numbers, function(q) {
  t_test <- t.test(df_final[[paste0("q_", q, "_llm")]], 
                   df_final[[paste0("q_", q, "_human")]], 
                   paired = TRUE, na.action = na.omit)
  
  # Convert test results to a tidy format
  tidy_result <- tidy(t_test)
  tidy_result$question <- as.integer(q)  # Store question number
  return(tidy_result)
}) %>%
  select(question, estimate, statistic, p.value, conf.low, conf.high)

# 3. F-test (ANOVA) for joint hypothesis (all question means are different)
df_long <- df_final %>%
  select(submission_id, all_of(paste0("q_", question_numbers, "_llm")), all_of(paste0("q_", question_numbers, "_human"))) %>%
  pivot_longer(
    cols = -submission_id,
    names_to = c("question", "source"),
    names_pattern = "q_(\\d+)_(.*)"
  ) %>%
  mutate(question = as.integer(question))

anova_test <- aov(value ~ as.factor(question) + source, data = df_long)
anova_summary <- summary(anova_test)

# Print results
print("T-test for Total Score Difference:")
print(t_test_total)

print("T-tests for Each Question's Score Difference:")
print(t_test_questions)

print("F-test for Joint Hypothesis (All Means Different):")
print(anova_summary)
