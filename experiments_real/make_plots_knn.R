# Install packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("xtable")) install.packages("xtable")

library(tidyverse)
library(xtable)

#-------------------------------------------------------------------------------
#                           Helper Functions
#-------------------------------------------------------------------------------
# Formats mean into a string (removed SE to save space)
format_metric <- function(val_mean) {
  if (length(val_mean) == 0) return("-")
  if (is.na(val_mean) || is.nan(val_mean)) return("-")
  sprintf("%.3f", val_mean)
}

#-------------------------------------------------------------------------------
#                           Table Generation Function
#-------------------------------------------------------------------------------
generate_combined_odor_table <- function(knn_results_dir, hdc_results_dir, save_path = NULL, target_alpha = 0.2, target_beta = 0.0) {
  
  rat_names <- c("Barat", "Buchanan", "Mitt", "Stella", "Superchris")
  
  #-------------------
  # 1. Load & Process KNN Data
  #-------------------
  knn_files <- list.files(path = knn_results_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(knn_files) == 0) stop("No KNN result files found! Check your directory path.")
  
  knn_data <- knn_files %>% map_df(~read_csv(., show_col_types = FALSE))
  knn_data <- knn_data %>%
    mutate(rat_name = factor(rat_id, levels = sort(unique(rat_id)), labels = rat_names))
  
  knn_summary <- knn_data %>%
    group_by(rat_name) %>%
    summarise(
      knn_cov = mean(point_acc, na.rm = TRUE),
      knn_size = 1.000,
      inv_cov = mean(set_cov, na.rm = TRUE),
      inv_size = mean(set_size, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    rowwise() %>%
    mutate(
      `KNN_Cov` = format_metric(knn_cov),
      `KNN_Size` = format_metric(knn_size),
      `KNN + Inv. quantile_Cov` = format_metric(inv_cov),
      `KNN + Inv. quantile_Size` = format_metric(inv_size)
    ) %>%
    ungroup() %>%
    select(rat_name, `KNN_Cov`, `KNN_Size`, `KNN + Inv. quantile_Cov`, `KNN + Inv. quantile_Size`) %>%
    pivot_longer(
      cols = -rat_name, 
      names_to = c("Method", "Metric"), 
      names_sep = "_", 
      values_to = "Value"
    ) %>%
    pivot_wider(names_from = Metric, values_from = Value)
  
  #-------------------
  # 2. Load & Process HDC Data
  #-------------------
  hdc_files <- list.files(path = hdc_results_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(hdc_files) == 0) stop("No HDC result files found! Check your directory path.")
  
  hdc_data <- hdc_files %>% map_df(~read_csv(., show_col_types = FALSE))
  hdc_data <- hdc_data %>%
    mutate(rat_name = factor(rat_id, levels = sort(unique(rat_id)), labels = rat_names))
  
  # Safely filter by target Alpha and Beta only if the columns exist
  if ("alpha" %in% colnames(hdc_data)) {
    hdc_data <- hdc_data %>% filter(is.na(alpha) | abs(alpha - target_alpha) < 1e-6)
  }
  if ("beta" %in% colnames(hdc_data)) {
    hdc_data <- hdc_data %>% filter(is.na(beta) | abs(beta - target_beta) < 1e-6)
  }
  
  # Apply HDC Singleton Logic 
  hdc_as_sets <- hdc_data %>%
    filter(score_type == "vanilla_full", exp == "point_valued") %>%
    select(random_state, rat_name, point_acc, any_of(c("alpha", "beta"))) %>%    
    mutate(
      exp = "set_valued",
      marginal = TRUE,
      score_type = "vanilla_full",
      set_cov = point_acc, # Coverage becomes Accuracy
      set_size = 1.0       # Size is always 1
    )
  
  hdc_data <- hdc_data %>%
    filter(!(score_type == "vanilla_full" & exp == "set_valued")) %>%
    bind_rows(hdc_as_sets)
  
  hdc_summary <- hdc_data %>%
    filter(exp == "set_valued", marginal == TRUE, score_type %in% c("vanilla_full", "inverse_quantile", "ratio")) %>%
    group_by(rat_name, score_type) %>%
    summarise(
      m_cov = mean(set_cov, na.rm = TRUE),
      m_size = mean(set_size, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    rowwise() %>%
    mutate(
      Method = case_when(
        score_type == "vanilla_full" ~ "HDC",
        score_type == "inverse_quantile" ~ "HDC + Inv. quantile",
        score_type == "ratio" ~ "CHDC-ratio"
      ),
      Cov = format_metric(m_cov),
      Size = format_metric(m_size)
    ) %>%
    ungroup() %>%
    select(rat_name, Method, Cov, Size)
  
  #-------------------
  # 3. Combine & Reshape
  #-------------------
  combined_long <- bind_rows(knn_summary, hdc_summary) %>%
    pivot_longer(cols = c(Cov, Size), names_to = "Metric", values_to = "Value")
  
  wide_data <- combined_long %>%
    mutate(ColName = paste(rat_name, Metric, sep = "_")) %>%
    select(-rat_name, -Metric) %>%
    pivot_wider(names_from = ColName, values_from = Value) %>%
    # Explicitly re-order the rows in the final table
    mutate(Method = factor(Method, levels = c("KNN", "KNN + Inv. quantile", "HDC", "HDC + Inv. quantile", "CHDC-ratio"))) %>%
    arrange(Method)
  
  #-------------------
  # 4. LaTeX Generation
  #-------------------
  align_str <- "l|l|cc|cc|cc|cc|cc|"
  
  additor <- list()
  # Add position 0 (top header) and 2 (after the second row, separating KNN and HDC)
  additor$pos <- list(0, 2) 
  additor$command <- c(
    paste0(
      "\\hline\n",
      " & \\multicolumn{2}{c|}{\\textbf{Barat}} & \\multicolumn{2}{c|}{\\textbf{Buchanan}} & \\multicolumn{2}{c|}{\\textbf{Mitt}} & \\multicolumn{2}{c|}{\\textbf{Stella}} & \\multicolumn{2}{c|}{\\textbf{Superchris}} \\\\\n",
      "\\cline{2-11}\n",
      "\\textbf{Method} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{Cov.} & \\textbf{Size} \\\\\n",
      "\\hline\n"
    ),
    "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n" # This line gets inserted after row 2 with spacing!
  )
  
  ltx <- xtable(wide_data, align = align_str)
  
  # Print/Save logic
  if (!is.null(save_path)) {
    if (!dir.exists(dirname(save_path))) dir.create(dirname(save_path), recursive = TRUE)
    print(ltx, file = save_path, floating = FALSE, include.rownames = FALSE, include.colnames = FALSE,
          sanitize.text.function = function(x) x, add.to.row = additor, hline.after = c(nrow(wide_data)), comment = FALSE)
    cat(paste("Saved Combined KNN + HDC table to:", save_path, "\n"))
  } else {
    print(ltx, floating = FALSE, include.rownames = FALSE, include.colnames = FALSE,
          sanitize.text.function = function(x) x, add.to.row = additor, hline.after = c(nrow(wide_data)), comment = FALSE)
  }
}

#-------------------------------------------------------------------------------
#                           Execution
#-------------------------------------------------------------------------------
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_real/results/")

knn_dir <- "./knn_odor_decoding/"
hdc_dir <- "./odor_decoding/"
out_file <- '../../results/tables/odor_decoding/knn_hdc_odor_decoding_results.tex'

# Generates the table and filters HDC by alpha = 0.2
generate_combined_odor_table(
  knn_results_dir = knn_dir, 
  hdc_results_dir = hdc_dir, 
  save_path = out_file,
  target_alpha = 0.2, 
  target_beta = 0.3
)