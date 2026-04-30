# Install packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("xtable")) install.packages("xtable")

library(tidyverse)
library(xtable)

#-------------------------------------------------------------------------------
#                           OOD AUROC Data Processing
#-------------------------------------------------------------------------------
# Set your base results directory here
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_ood/")
base_dir <- "./results/"

# Define the experiment folders and their corresponding table headers
exp_folders <- c("ood_isolet", "ood_mnist", "ood_languages", "ood_har")
exp_labels  <- c("ISOLET", "MNIST", "Languages", "UCI HAR")

# Helper function to load and summarize a single dataset
process_dataset <- function(folder_name, col_name) {
  path <- file.path(base_dir, folder_name)
  files <- list.files(path = path, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(files) == 0) {
    warning(paste("No files found in", path))
    # Return a dummy dataframe if missing so the join doesn't break
    return(tibble(method = character(), !!sym(col_name) := character()))
  }
  
  # Read all CSVs
  df <- files %>% map_df(~read_csv(., show_col_types = FALSE))
  
  # Calculate Mean and SE, format as "Mean (SE)"
  summary_df <- df %>%
    group_by(method) %>%
    summarise(
      n = n(),
      mean_val = mean(ood_auroc, na.rm = TRUE),
      se_val = sd(ood_auroc, na.rm = TRUE) / sqrt(n),
      .groups = "drop"
    ) %>%
    mutate(
      !!sym(col_name) := sprintf("%.3f (%.3f)", mean_val, se_val)
    ) %>%
    select(method, !!sym(col_name))
  
  return(summary_df)
}

# Process all datasets and merge them by method
all_summaries <- map2(exp_folders, exp_labels, process_dataset)
combined_df <- reduce(all_summaries, full_join, by = "method")

# Rename methods and reorder rows
final_tab <- combined_df %>%
  mutate(Method = case_when(
    method == "chdc_sim"      ~ "CP-similarity",
    method == "chdc_discount" ~ "CP-discount",
    method == "maxsim"        ~ "Max similarity",
    method == "energy"        ~ "Energy",
    TRUE                      ~ method
  )) %>%
  filter(Method %in% c("CP-similarity", "CP-discount", "Max similarity", "Energy")) %>%
  arrange(factor(Method, levels = c("CP-similarity", "CP-discount", "Max similarity", "Energy"))) %>%
  select(Method, all_of(exp_labels))

# Replace any missing data (NA) with a dash
final_tab[is.na(final_tab)] <- "-"


#----------------------------------------
#             LaTeX Table 
#----------------------------------------
generate_table <- function(final_tab, save_path = NULL) {
  # Structure: Method (left) | 4 Data columns (center)
  align_str <- "l|l|cccc|"
  
  # Construct the custom header
  additor <- list(
    pos = list(0),
    command = paste0(
      "\\hline\n",
      "\\textbf{Method} & \\textbf{ISOLET} & \\textbf{MNIST} & \\textbf{Languages} & \\textbf{UCI HAR} \\\\\n",
      "\\hline\n"
    )
  )
  
  ltx <- xtable(final_tab, align = align_str)
  
  # Print/Save
  if (!is.null(save_path)) {
    if (!dir.exists(dirname(save_path))) dir.create(dirname(save_path), recursive = TRUE)
    
    print(ltx, 
          file = save_path, 
          floating = FALSE, 
          include.rownames = FALSE, 
          include.colnames = FALSE,
          sanitize.text.function = function(x) x, 
          add.to.row = additor, 
          hline.after = c(nrow(final_tab)), 
          comment = FALSE)
    
    cat(paste("\nSaved combined tabular to:", save_path, "\n"))
  } else {
    print(ltx, 
          floating = FALSE, 
          include.rownames = FALSE, 
          include.colnames = FALSE,
          sanitize.text.function = function(x) x, 
          add.to.row = additor, 
          hline.after = c(nrow(final_tab)), 
          comment = FALSE)
  }
}

#------------------
# Execution
#------------------
table_dir <- '../results/tables/'
out_file <- file.path(table_dir, "ood_summary_table.tex")
generate_table(final_tab, save_path = out_file)