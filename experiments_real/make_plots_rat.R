# Install packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("xtable")) install.packages("xtable")
if (!require("stringr")) install.packages("stringr")

library(tidyverse)
library(xtable)
library(stringr)

#-------------------------------------------------------------------------------
#                           Odor Decoding Exp
#-------------------------------------------------------------------------------
# Set your results directory here
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_real/results/")
results_dir <- "./odor_decoding/" 

# Pattern matches: ratX_seedY_alphaZ_betaW.csv
files <- list.files(path = results_dir, pattern = "rat.*_seed.*_alpha.*_beta.*\\.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found! Check your directory path.")
} else {
  full_data <- files %>% 
    map_df(~read_csv(., show_col_types = FALSE))
  cat(paste("Loaded", length(files), "files. Total rows:", nrow(full_data), "\n"))
}

#------------------
# Table Generation 
#------------------
format_metric <- function(val_mean, val_se) {
  if (is.na(val_mean)) return("-")
  sprintf("%.3f (%.3f)", val_mean, val_se)
}

create_odor_full_table <- function(df, target_alpha, target_beta, save_dir = NULL) {
  
  # Map rat_id to names
  rat_names <- c('Barat', 'Buchanan', 'Mitt', 'Stella', 'Superchris')
  
  # Filter Data (include NA alpha/beta for the training baseline)
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha),
           abs(beta - target_beta) < 1e-6 | is.na(beta)) %>%
    mutate(Rat_Full = rat_names[rat_id + 1])
  
  if (nrow(dat) == 0) {
    warning("No data found for these parameters.")
    return(NULL)
  }
  
  # 3. HDC Singleton Logic (Treat Point Acc as Coverage for Vanilla/Train)
  hdc_as_sets <- dat %>%
    filter(score_type %in% c("vanilla_full", "vanilla_train"), exp == "point_valued") %>%
    mutate(
      exp = "set_valued",
      marginal = TRUE,
      set_cov = point_acc,
      set_size = 1.0
    )
  
  # Merge back and clean up existing set_valued rows for these specific types
  dat <- dat %>%
    filter(!(score_type %in% c("vanilla_full", "vanilla_train") & exp == "set_valued")) %>%
    bind_rows(hdc_as_sets)
  
  # Aggregation
  set_metrics <- dat %>%
    filter(exp == "set_valued", marginal == TRUE) %>%
    group_by(Rat_Full, score_type) %>%
    summarise(
      n = n(),
      cov_u = mean(set_cov, na.rm=TRUE), cov_se = sd(set_cov, na.rm=TRUE)/sqrt(n),
      siz_u = mean(set_size, na.rm=TRUE), siz_se = sd(set_size, na.rm=TRUE)/sqrt(n),
      .groups = "drop"
    )
  
  point_metrics <- dat %>%
    filter(exp == "point_valued") %>%
    group_by(Rat_Full, score_type) %>%
    summarise(
      n = n(),
      acc_u = mean(point_acc, na.rm=TRUE), acc_se = sd(point_acc, na.rm=TRUE)/sqrt(n),
      .groups = "drop"
    )
  
  ood_metrics <- dat %>%
    filter(exp == "ood", marginal == TRUE) %>%
    group_by(Rat_Full, score_type) %>%
    summarise(
      n = n(),
      ood_u = mean(ood_auroc, na.rm=TRUE), ood_se = sd(ood_auroc, na.rm=TRUE)/sqrt(n),
      .groups = "drop"
    )
  
  # 5. Merge & Format
  formatted <- set_metrics %>%
    left_join(point_metrics, by=c("Rat_Full", "score_type")) %>%
    left_join(ood_metrics, by=c("Rat_Full", "score_type")) %>%
    mutate(
      Cov = mapply(format_metric, cov_u, cov_se),
      Size = mapply(format_metric, siz_u, siz_se),
      Accuracy = mapply(format_metric, acc_u, acc_se),
      AUROC = mapply(format_metric, ood_u, ood_se),
      Method = case_when(
        score_type == "vanilla_full"  ~ "HDC",
        score_type == "vanilla_train" ~ "HDC (train)",
        score_type == "inverse_quantile" ~ "Inv. quantile",
        score_type == "penalized" ~ "Penalized",
        score_type == "sim" ~ "Similarity",
        score_type == "ratio" ~ "CHDC-ratio",
        score_type == "discount" ~ "CHDC-discount",
        TRUE ~ score_type
      )
    ) %>%
    # Use exact reordering logic
    arrange(factor(Rat_Full, levels = rat_names), 
            factor(Method, levels = c("HDC (train)", "HDC", "Inv. quantile", 
                                      "Penalized", "Similarity", "CHDC-ratio", "CHDC-discount")))
  
  final_df <- formatted %>%
    group_by(Rat_Full) %>%
    mutate(Rat_Display = if_else(
      row_number() == 1, 
      paste0("\\multirow{", n(), "}{*}{", Rat_Full, "}"), 
      ""
    )) %>%
    ungroup() %>%
    select(Rat_Display, Method, Cov, Size, Accuracy, AUROC) %>%
    rename(Rat = Rat_Display)
  
  # Align: Rat | Method | Cov Size | Acc | OOD
  align_str <- "l|ll|cc|c|c|"
  
  # Grouping lines between rats (similar to sigma groups)
  rat_rle <- rle(as.character(formatted$Rat_Full))
  hlines <- cumsum(rat_rle$lengths)
  
  # Header Construction: Move "Rat" and "Method" to the second row
  additor <- list(pos = list(0), command = paste0(
    "\\hline\n",
    " & & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
    "\\cline{3-6}\n",
    "\\textbf{Rat} & \\textbf{Method} & \\textbf{Coverage} & \\textbf{Size} & \\textbf{Accuracy} & \\textbf{AUC} \\\\\n",
    "\\hline\n"
  ))
  
  ltx <- xtable(final_df, align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filename <- file.path(save_dir, paste0("odor_decoding_full_alpha", target_alpha, ".tex"))
    
    print(ltx, file = filename, 
          floating = FALSE, 
          include.rownames = FALSE, 
          include.colnames = FALSE, 
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = hlines, 
          comment = FALSE)
    cat(paste("Saved full table to:", filename, "\n"))
  } else {
    print(ltx, floating = FALSE, include.rownames = FALSE, include.colnames = FALSE,
          sanitize.text.function = function(x){x}, add.to.row = additor, 
          hline.after = hlines, comment = FALSE)
  }
}

create_odor_summary_table <- function(df, target_alpha, target_beta, save_dir = NULL) {
  
  # 1. Define targets and mapping
  target_rats <- c('Buchanan', 'Stella', 'Superchris')
  rat_names_map <- c('Barat', 'Buchanan', 'Mitt', 'Stella', 'Superchris')
  
  # 2. Filter and Process Data
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha),
           abs(beta - target_beta) < 1e-6 | is.na(beta)) %>%
    mutate(Rat_Full = rat_names_map[rat_id + 1]) %>%
    filter(Rat_Full %in% target_rats)
  
  # Treat Point Acc as Coverage for Vanilla HDC
  hdc_as_sets <- dat %>%
    filter(score_type == "vanilla_full", exp == "point_valued") %>%
    mutate(exp = "set_valued", marginal = TRUE, set_cov = point_acc, set_size = 1.0)
  
  dat <- dat %>%
    filter(!(score_type == "vanilla_full" & exp == "set_valued")) %>%
    bind_rows(hdc_as_sets)
  
  # 3. Aggregate Means
  summary_stats <- dat %>%
    group_by(Rat_Full, score_type) %>%
    summarise(
      Cov = mean(set_cov[exp == "set_valued" & marginal == TRUE], na.rm=TRUE),
      Size = mean(set_size[exp == "set_valued" & marginal == TRUE], na.rm=TRUE),
      AUC = mean(ood_auroc[exp == "ood" & marginal == TRUE], na.rm=TRUE),
      .groups = "drop"
    ) %>%
    mutate(Method = case_when(
      score_type == "vanilla_full"  ~ "HDC",
      score_type == "inverse_quantile" ~ "Inv. quantile",
      score_type == "penalized" ~ "Penalized",
      score_type == "sim" ~ "Similarity",
      score_type == "ratio" ~ "CHDC-ratio",
      score_type == "discount" ~ "CHDC-discount",
      TRUE ~ score_type
      ),
      across(c(Cov, Size, AUC), ~ ifelse(is.na(.), "-", sprintf("%.3f", .)))
    ) %>%
    filter(score_type != "vanilla_train") # Omit HDC (train)
  
  # Pivot wider so each rat has its own set of 3 columns
  wide_df <- summary_stats %>%
    select(Method, Rat_Full, Cov, Size, AUC) %>%
    pivot_wider(names_from = Rat_Full, values_from = c(Cov, Size, AUC), names_glue = "{Rat_Full}_{.value}") %>%
    arrange(factor(Method, levels = c("HDC", "Inv. quantile", "Penalized", "Similarity", "CHDC-ratio", "CHDC-discount")))
  
  # Reorder columns to group by Rat: [Method, Rat1_Cov, Rat1_Size, Rat1_AUC, Rat2_Cov...]
  col_order <- c("Method", as.vector(t(outer(target_rats, c("_Cov", "_Size", "_AUC"), paste0))))
  wide_df <- wide_df[, col_order]
  
  # LaTeX Formatting
  align_str <- "l|l|ccc|ccc|ccc|"
  
  # Custom Header Construction
  header_cmd <- paste0(
    "\\hline\n",
    " & \\multicolumn{3}{c|}{\\textbf{Buchanan}} & \\multicolumn{3}{c|}{\\textbf{Stella}} & \\multicolumn{3}{c|}{\\textbf{Superchris}} \\\\\n",
    "\\cline{2-10}\n",
    "\\textbf{Method} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} \\\\\n",
    "\\hline\n"
  )
  
  ltx <- xtable(wide_df, align = align_str)
  
  print(ltx, 
        file = if(!is.null(save_dir)) file.path(save_dir, "odor_decoding_summary_table.tex") else "",
        floating = FALSE, 
        include.rownames = FALSE, 
        include.colnames = FALSE, 
        sanitize.text.function = function(x) x,
        add.to.row = list(pos = list(0), command = header_cmd),
        hline.after = c(nrow(wide_df)),
        comment = FALSE)
}


#------------------
# Execution
#------------------
table_dir <- '../../results/tables/odor_decoding/'

#create_odor_full_table(full_data, target_alpha = 0.2, target_beta = 0.3, save_dir = table_dir)

create_odor_summary_table(full_data, target_alpha = 0.2, target_beta = 0.3, save_dir = table_dir)
