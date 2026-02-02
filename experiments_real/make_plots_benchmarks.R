# Install packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("xtable")) install.packages("xtable")
if (!require("stringr")) install.packages("stringr")
if (!require("gridExtra")) install.packages("gridExtra")

library(tidyverse)
library(xtable)
library(stringr)
library(gridExtra)

#-------------------------------------------------------------------------------
#                           Isolet Exp
#-------------------------------------------------------------------------------
# Set your results directory here
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_real/results/")
results_dir <- "./isolet/" 

# Pattern matches: seedX_alphaY.csv
files <- list.files(path = results_dir, pattern = "seed.*_alpha.*\\.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found! Check your directory path.")
} else {
  isolet_data <- files %>% 
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

create_isolet_table <- function(df, target_alpha, save_dir = NULL) {
  
  # 1. Filter Data by Alpha (keep NA alphas for Vanilla baselines if needed)
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha)) %>%
    filter(score_type != "vanilla_train")
  
  if (nrow(dat) == 0) {
    warning(paste("No data found for alpha =", target_alpha))
    return(NULL)
  }
  
  #-----------------------
  #  HDC Singleton Logic
  #-----------------------
  # Treat HDC (vanilla_full) as producing a singleton set:
  #   - Set Coverage = Point Accuracy
  #   - Set Size     = 1.0
  
  hdc_as_sets <- dat %>%
    filter(score_type == "vanilla_full", exp == "point_valued") %>%
    select(random_state, point_acc, any_of(c("alpha", "sigma"))) %>%    
    mutate(
      exp = "set_valued",
      marginal = TRUE,
      score_type = "vanilla_full",
      set_cov = point_acc, # Coverage becomes Accuracy
      set_size = 1.0       # Size is always 1
    )
  
  # Remove any existing "set_valued" rows for vanilla_full and bind the new logic
  dat <- dat %>%
    filter(!(score_type == "vanilla_full" & exp == "set_valued")) %>%
    bind_rows(hdc_as_sets)
  
  #------------------------------------------------
  # Aggregation (Grouped by Score Type)
  #------------------------------------------------
  
  # --- A. Set-Valued Metrics (Marginal Only) ---
  set_metrics <- dat %>%
    filter(exp == "set_valued", marginal == TRUE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      m_cov_u = mean(set_cov, na.rm=TRUE), 
      m_cov_se = sd(set_cov, na.rm=TRUE) / sqrt(n),
      m_siz_u = mean(set_size, na.rm=TRUE), 
      m_siz_se = sd(set_size, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- B. Point-Valued Metrics ---
  point_metrics <- dat %>%
    filter(exp == "point_valued") %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      acc_u = mean(point_acc, na.rm=TRUE), 
      acc_se = sd(point_acc, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- C. OOD Metrics ---
  ood_metrics <- dat %>%
    filter(exp == "ood", marginal == TRUE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      ood_u = mean(ood_auroc, na.rm=TRUE), 
      ood_se = sd(ood_auroc, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- D. Merge & Format ---
  base <- tibble(score_type = unique(dat$score_type))
  
  joined <- base %>%
    left_join(set_metrics, by="score_type") %>%
    left_join(point_metrics, by="score_type") %>%
    left_join(ood_metrics, by="score_type")
  
  formatted <- joined %>%
    mutate(
      `Cov` = mapply(format_metric, m_cov_u, m_cov_se),
      `Size` = mapply(format_metric, m_siz_u, m_siz_se),
      `Accuracy` = mapply(format_metric, acc_u, acc_se),
      `AUROC` = mapply(format_metric, ood_u, ood_se)
    )
  
  # --- E. Rename & Reorder (Strict 3-Class Convention) ---
  formatted <- formatted %>%
    mutate(Method = case_when(
      score_type == "vanilla_full" ~ "HDC",
      score_type == "inverse_quantile" ~ "Inv. quantile",
      score_type == "penalized" ~ "Penalized",
      score_type == "sim" ~ "Similarity",
      score_type == "ratio" ~ "CHDC-ratio",
      score_type == "discount" ~ "CHDC-discount",
      TRUE ~ score_type
    )) %>%
    filter(Method != "vanilla_train") %>%
    # Use exact same order as 3-class
    arrange(factor(Method, levels = c("HDC", "Inv. quantile", "Penalized", 
                                      "Similarity", "CHDC-ratio", "CHDC-discount"))) %>%
    select(Method, Cov, Size, Accuracy, AUROC)
  
  # --- F. Construct LaTeX Header ---
  # Structure: Method | Cov Size | Acc | OOD |
  align_str <- "l|l|cc|c|c|"
  
  additor <- list()
  additor$pos <- list(0)
  additor$command <- paste0(
    "\\hline\n",
    " & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
    "\\cline{2-5}\n",
    "\\textbf{Method} & \\textbf{Coverage} & \\textbf{Size} & \\textbf{Accuracy} & \\textbf{AUC} \\\\\n",
    "\\hline\n"
  )
  
  # --- G. Save/Print ---
  # floating = FALSE ensures we only get the tabular environment
  ltx <- xtable(formatted, align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filename <- file.path(save_dir, paste0("isolet_table_alpha", target_alpha, ".tex"))
    
    print(ltx, file = filename, 
          floating = FALSE,          # No \begin{table} wrapper
          include.rownames = FALSE, 
          include.colnames = FALSE, 
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = c(nrow(formatted)),
          comment = FALSE)
    
    cat(paste("Saved tabular to:", filename, "\n"))
  } else {
    print(ltx, 
          floating = FALSE,
          include.rownames = FALSE, 
          include.colnames = FALSE,
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = c(nrow(formatted)),
          comment = FALSE)
  }
}

#------------------
# Save Table 
#------------------
table_dir <- '../../results/tables/isolet/'

create_isolet_table(isolet_data, target_alpha = 0.05, save_dir = table_dir)
create_isolet_table(isolet_data, target_alpha = 0.02, save_dir = table_dir)
create_isolet_table(isolet_data, target_alpha = 0.01, save_dir = table_dir)





#-------------------------------------------------------------------------------
#                           MNIST Exp
#-------------------------------------------------------------------------------
# Set your results directory here
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_real/results/")
results_dir <- "./mnist/" 

# Pattern matches: seedX_alphaY.csv
files <- list.files(path = results_dir, pattern = "seed.*_alpha.*\\.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found! Check your directory path.")
} else {
  mnist_data <- files %>% 
    map_df(~read_csv(., show_col_types = FALSE))
  cat(paste("Loaded", length(files), "files. Total rows:", nrow(mnist_data), "\n"))
}

#------------------
# Table Generation 
#------------------
format_metric <- function(val_mean, val_se) {
  if (is.na(val_mean)) return("-")
  sprintf("%.3f (%.3f)", val_mean, val_se)
}

create_mnist_table <- function(df, target_alpha, save_dir = NULL) {
  
  # 1. Filter Data by Alpha (keep NA alphas for Vanilla baselines if needed)
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha)) %>%
    filter(score_type != "vanilla_train")
  
  if (nrow(dat) == 0) {
    warning(paste("No data found for alpha =", target_alpha))
    return(NULL)
  }
  
  #-----------------------
  #  HDC Singleton Logic
  #-----------------------
  # Treat HDC (vanilla_full) as producing a singleton set:
  #   - Set Coverage = Point Accuracy
  #   - Set Size     = 1.0
  
  hdc_as_sets <- dat %>%
    filter(score_type == "vanilla_full", exp == "point_valued") %>%
    select(random_state, point_acc, any_of(c("alpha", "sigma"))) %>%    
    mutate(
      exp = "set_valued",
      marginal = TRUE,
      score_type = "vanilla_full",
      set_cov = point_acc, # Coverage becomes Accuracy
      set_size = 1.0       # Size is always 1
    )
  
  # Remove any existing "set_valued" rows for vanilla_full and bind the new logic
  dat <- dat %>%
    filter(!(score_type == "vanilla_full" & exp == "set_valued")) %>%
    bind_rows(hdc_as_sets)
  
  #------------------------------------------------
  # Aggregation (Grouped by Score Type)
  #------------------------------------------------
  
  # --- A. Set-Valued Metrics (Marginal Only) ---
  set_metrics <- dat %>%
    filter(exp == "set_valued", marginal == TRUE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      m_cov_u = mean(set_cov, na.rm=TRUE), 
      m_cov_se = sd(set_cov, na.rm=TRUE) / sqrt(n),
      m_siz_u = mean(set_size, na.rm=TRUE), 
      m_siz_se = sd(set_size, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- B. Point-Valued Metrics ---
  point_metrics <- dat %>%
    filter(exp == "point_valued") %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      acc_u = mean(point_acc, na.rm=TRUE), 
      acc_se = sd(point_acc, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- C. OOD Metrics ---
  ood_metrics <- dat %>%
    filter(exp == "ood", marginal == TRUE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      ood_u = mean(ood_auroc, na.rm=TRUE), 
      ood_se = sd(ood_auroc, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- D. Merge & Format ---
  base <- tibble(score_type = unique(dat$score_type))
  
  joined <- base %>%
    left_join(set_metrics, by="score_type") %>%
    left_join(point_metrics, by="score_type") %>%
    left_join(ood_metrics, by="score_type")
  
  formatted <- joined %>%
    mutate(
      `Cov` = mapply(format_metric, m_cov_u, m_cov_se),
      `Size` = mapply(format_metric, m_siz_u, m_siz_se),
      `Accuracy` = mapply(format_metric, acc_u, acc_se),
      `AUROC` = mapply(format_metric, ood_u, ood_se)
    )
  
  # --- E. Rename & Reorder (Strict 3-Class Convention) ---
  formatted <- formatted %>%
    mutate(Method = case_when(
      score_type == "vanilla_full" ~ "HDC",
      score_type == "inverse_quantile" ~ "Inv. quantile",
      score_type == "penalized" ~ "Penalized",
      score_type == "sim" ~ "Similarity",
      score_type == "ratio" ~ "CHDC-ratio",
      score_type == "discount" ~ "CHDC-discount",
      TRUE ~ score_type
    )) %>%
    filter(Method != "vanilla_train") %>%
    # Use exact same order as 3-class/Isolet
    arrange(factor(Method, levels = c("HDC", "Inv. quantile", "Penalized", 
                                      "Similarity", "CHDC-ratio", "CHDC-discount"))) %>%
    select(Method, Cov, Size, Accuracy, AUROC)
  
  # --- F. Construct LaTeX Header ---
  # Structure: Method | Cov Size | Acc | OOD |
  # Align: 'll' (rownames + Method) | 'cc' | 'c' | 'c' |
  align_str <- "l|l|cc|c|c|"
  
  additor <- list()
  additor$pos <- list(0)
  additor$command <- paste0(
    "\\hline\n",
    " & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
    "\\cline{2-5}\n",
    "\\textbf{Method} & \\textbf{Coverage} & \\textbf{Size} & \\textbf{Accuracy} & \\textbf{AUC} \\\\\n",
    "\\hline\n"
  )
  
  # --- G. Save/Print ---
  ltx <- xtable(formatted, align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filename <- file.path(save_dir, paste0("mnist_table_alpha", target_alpha, ".tex"))
    
    print(ltx, file = filename, 
          floating = FALSE,          
          include.rownames = FALSE, 
          include.colnames = FALSE, 
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = c(nrow(formatted)),
          comment = FALSE)
    
    cat(paste("Saved tabular to:", filename, "\n"))
  } else {
    print(ltx, 
          floating = FALSE,
          include.rownames = FALSE, 
          include.colnames = FALSE,
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = c(nrow(formatted)),
          comment = FALSE)
  }
}

#------------------
# Save Table 
#------------------
table_dir <- '../../results/tables/mnist/'

# Generate for typical Alphas
create_mnist_table(mnist_data, target_alpha = 0.05, save_dir = table_dir)
create_mnist_table(mnist_data, target_alpha = 0.1,  save_dir = table_dir)



#-------------------------------------------------------------------------------
#                 Combined Summary Table (Mixed Alphas)
#-------------------------------------------------------------------------------

create_combined_table <- function(iso_df, mnist_df, iso_alpha, mnist_alpha, save_path = NULL) {
  
  get_summary <- function(df, target_alpha) {
    dat <- df %>% 
      filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha)) %>%
      filter(score_type != "vanilla_train")
    
    hdc_pts <- dat %>%
      filter(score_type == "vanilla_full", exp == "point_valued") %>%
      mutate(exp = "set_valued", marginal = TRUE, score_type = "vanilla_full",
             set_cov = point_acc, set_size = 1.0)
    
    dat <- dat %>% filter(!(score_type == "vanilla_full" & exp == "set_valued")) %>% bind_rows(hdc_pts)
    
    dat %>%
      group_by(score_type) %>%
      summarise(
        Cov = ifelse(all(is.na(set_cov[exp == "set_valued" & marginal])), "-", 
                     sprintf("%.3f", mean(set_cov[exp == "set_valued" & marginal], na.rm=TRUE))),
        Size = ifelse(all(is.na(set_size[exp == "set_valued" & marginal])), "-", 
                      sprintf("%.3f", mean(set_size[exp == "set_valued" & marginal], na.rm=TRUE))),
        AUROC = ifelse(all(is.na(ood_auroc[exp == "ood" & marginal])), "-", 
                       sprintf("%.3f", mean(ood_auroc[exp == "ood" & marginal], na.rm=TRUE))),
        .groups = "drop"
      ) %>%
      mutate(Method = case_when(
        score_type == "vanilla_full" ~ "HDC",
        score_type == "inverse_quantile" ~ "Inv. quantile",
        score_type == "penalized" ~ "Penalized",
        score_type == "sim" ~ "Similarity",
        score_type == "ratio" ~ "CHDC-ratio",
        score_type == "discount" ~ "CHDC-discount",
        TRUE ~ score_type
      ))
  }
  
  iso_summ <- get_summary(iso_df, iso_alpha)
  mni_summ <- get_summary(mnist_df, mnist_alpha)
  
  final_tab <- iso_summ %>%
    select(Method, Cov_Iso = Cov, Size_Iso = Size, AUROC_Iso = AUROC) %>%
    full_join(
      mni_summ %>% select(Method, Cov_Mni = Cov, Size_Mni = Size, AUROC_Mni = AUROC),
      by = "Method"
    ) %>%
    arrange(factor(Method, levels = c("HDC", "Inv. quantile", "Penalized", 
                                      "Similarity", "CHDC-ratio", "CHDC-discount")))
  
  align_str <- "l|l|ccc|ccc|"
  
  iso_head <- paste0("\\textbf{ISOLET ($\\alpha=", iso_alpha, "$)}")
  mni_head <- paste0("\\textbf{MNIST ($\\alpha=", mnist_alpha, "$)}")
  
  additor <- list(pos = list(0), command = paste0(
    "\\hline\n",
    " & \\multicolumn{3}{c|}{", iso_head, "} & \\multicolumn{3}{c|}{", mni_head, "} \\\\\n",
    "\\cline{2-7}\n",
    "\\textbf{Method} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} \\\\\n",
    "\\hline\n"
  ))
  
  ltx <- xtable(final_tab, align = align_str)
  
  print_args <- list(
    x = ltx,
    floating = FALSE,
    include.rownames = FALSE,
    include.colnames = FALSE,
    sanitize.text.function = function(x) x,
    add.to.row = additor,
    hline.after = c(nrow(final_tab)),
    comment = FALSE
  )
  
  if (!is.null(save_path)) {
    if (!dir.exists(dirname(save_path))) dir.create(dirname(save_path), recursive = TRUE)
    do.call(print, c(print_args, list(file = save_path)))
    cat(paste("Saved tabular to:", save_path, "\n"))
  } else {
    do.call(print, print_args)
  }
}

# Generate Table
combined_path <- '../../results/tables/benchmark_dataset_summary.tex'
create_combined_table(isolet_data, mnist_data, iso_alpha = 0.02, mnist_alpha = 0.05, save_path = combined_path)