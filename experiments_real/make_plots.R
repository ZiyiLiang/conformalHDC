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
  cat(paste("Loaded", length(files), "files. Total rows:", nrow(isolet_data), "\n"))
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
#create_isolet_table(isolet_data, target_alpha = 0.01, save_dir = table_dir)





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
  
  # Filter Data by Alpha 
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
#                         Languages Exp
#-------------------------------------------------------------------------------
# Set your results directory here
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_real/results/")
results_dir <- "./languages/" 

# Pattern matches: seedX_alphaY.csv
files <- list.files(path = results_dir, pattern = "seed.*_alpha.*\\.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found! Check your directory path.")
} else {
  languages_data <- files %>% 
    map_df(~read_csv(., show_col_types = FALSE))
  cat(paste("Loaded", length(files), "files. Total rows:", nrow(languages_data), "\n"))
}

#------------------
# Table Generation 
#------------------
format_metric <- function(val_mean, val_se) {
  if (is.na(val_mean)) return("-")
  sprintf("%.3f (%.4f)", val_mean, val_se)
}

create_languages_table <- function(df, target_alpha, save_dir = NULL) {
  
  # Filter Data by Alpha
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha)) %>%
    filter(score_type != "vanilla_train")
  
  if (nrow(dat) == 0) {
    warning(paste("No data found for alpha =", target_alpha))
    return(NULL)
  }
  
  #-----------------------
  #   HDC Singleton Logic
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
  ltx <- xtable(formatted, align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filename <- file.path(save_dir, paste0("languages_table_alpha", target_alpha, ".tex"))
    
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
table_dir <- '../../results/tables/languages/'
create_languages_table(languages_data, target_alpha = 0.01, save_dir = table_dir)



#-------------------------------------------------------------------------------
#                           Combined Summary Table 
#-------------------------------------------------------------------------------
create_combined_table <- function(iso_df, mnist_df, lang_df, 
                                  iso_alpha, mnist_alpha, lang_alpha, 
                                  save_path = NULL) {
  
  # Internal helper to aggregate metrics for a specific dataset and alpha
  get_summary <- function(df, target_alpha) {
    dat <- df %>% 
      filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha)) %>%
      filter(score_type != "vanilla_train")
    
    # HDC Singleton Logic: Convert point accuracy to set coverage/size
    hdc_pts <- dat %>%
      filter(score_type == "vanilla_full", exp == "point_valued") %>%
      mutate(exp = "set_valued", marginal = TRUE, score_type = "vanilla_full",
             set_cov = point_acc, set_size = 1.0)
    
    dat <- dat %>% 
      filter(!(score_type == "vanilla_full" & exp == "set_valued")) %>% 
      bind_rows(hdc_pts)
    
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
  
  # Generate Summaries
  iso_summ <- get_summary(iso_df, iso_alpha)
  mni_summ <- get_summary(mnist_df, mnist_alpha)
  lng_summ <- get_summary(lang_df, lang_alpha)
  
  # Join datasets: MNIST, Languages, ISOLET
  final_tab <- mni_summ %>%
    select(Method, Cov_Mni = Cov, Size_Mni = Size, AUROC_Mni = AUROC) %>%
    full_join(
      lng_summ %>% select(Method, Cov_Lng = Cov, Size_Lng = Size, AUROC_Lng = AUROC),
      by = "Method"
    ) %>%
    full_join(
      iso_summ %>% select(Method, Cov_Iso = Cov, Size_Iso = Size, AUROC_Iso = AUROC),
      by = "Method"
    ) %>%
    arrange(factor(Method, levels = c("HDC", "Inv. quantile", "Penalized", 
                                      "Similarity", "CHDC-ratio", "CHDC-discount")))
  
  # LaTeX Formatting
  # Alignment: Method (l) | MNIST (ccc) | Languages (ccc) | ISOLET (ccc)
  align_str <- "l|l|ccc|ccc|ccc|"
  
  mni_head <- paste0("\\textbf{MNIST ($\\alpha=", mnist_alpha, "$)}")
  lng_head <- paste0("\\textbf{Languages ($\\alpha=", lang_alpha, "$)}")
  iso_head <- paste0("\\textbf{ISOLET ($\\alpha=", iso_alpha, "$)}")
  
  additor <- list(pos = list(0), command = paste0(
    "\\hline\n",
    " & \\multicolumn{3}{c|}{", mni_head, "} & \\multicolumn{3}{c|}{", lng_head, "} & \\multicolumn{3}{c|}{", iso_head, "} \\\\\n",
    "\\cline{2-10}\n",
    "\\textbf{Method} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} \\\\\n",
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
  
  # Save/Print
  if (!is.null(save_path)) {
    if (!dir.exists(dirname(save_path))) dir.create(dirname(save_path), recursive = TRUE)
    do.call(print, c(print_args, list(file = save_path)))
    cat(paste("Saved combined tabular to:", save_path, "\n"))
  } else {
    do.call(print, print_args)
  }
}

#------------------
# Generate Table
#------------------
combined_path <- '../../results/tables/benchmark_dataset_summary.tex'

create_combined_table(
  iso_df = isolet_data, 
  mnist_df = mnist_data, 
  lang_df = languages_data, 
  iso_alpha = 0.02, 
  mnist_alpha = 0.05, 
  lang_alpha = 0.01, 
  save_path = combined_path
)

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
  odor_data <- files %>% 
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



#-------------------------------------------------------------------------------
#                    Combined Summary Table (with Odor Exp)
#-------------------------------------------------------------------------------
create_combined_with_rat_table <- function(iso_df, mnist_df, lang_df, odor_df,
                                           iso_alpha, mnist_alpha, lang_alpha, 
                                           rat_name, rat_alpha, rat_beta,
                                           save_path = NULL) {
  
  # Internal helper to aggregate metrics (reused from your existing logic)
  get_summary <- function(df, target_alpha, target_beta = NULL) {
    # Filter for alpha and beta (if beta is provided)
    dat <- df %>% 
      filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha))
    
    if (!is.null(target_beta)) {
      dat <- dat %>% filter(abs(beta - target_beta) < 1e-6 | is.na(beta))
    }
    
    dat <- dat %>% filter(score_type != "vanilla_train")
    
    # HDC Singleton Logic
    hdc_pts <- dat %>%
      filter(score_type == "vanilla_full", exp == "point_valued") %>%
      mutate(exp = "set_valued", marginal = TRUE, score_type = "vanilla_full",
             set_cov = point_acc, set_size = 1.0)
    
    dat <- dat %>% 
      filter(!(score_type == "vanilla_full" & exp == "set_valued")) %>% 
      bind_rows(hdc_pts)
    
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
  
  # Prepare Dataset Summaries
  iso_summ <- get_summary(iso_df, iso_alpha)
  mni_summ <- get_summary(mnist_df, mnist_alpha)
  lng_summ <- get_summary(lang_df, lang_alpha)
  
  # Prepare Specific Rat Summary
  rat_map <- c('Barat', 'Buchanan', 'Mitt', 'Stella', 'Superchris')
  rat_data_filtered <- odor_df %>% 
    mutate(Rat_Full = rat_map[rat_id + 1]) %>%
    filter(Rat_Full == rat_name)
  
  rat_summ <- get_summary(rat_data_filtered, rat_alpha, rat_beta)
  
  # Join all 4 sources
  final_tab <- mni_summ %>%
    select(Method, Cov_Mni = Cov, Size_Mni = Size, AUROC_Mni = AUROC) %>%
    full_join(lng_summ %>% select(Method, Cov_Lng = Cov, Size_Lng = Size, AUROC_Lng = AUROC), by = "Method") %>%
    full_join(iso_summ %>% select(Method, Cov_Iso = Cov, Size_Iso = Size, AUROC_Iso = AUROC), by = "Method") %>%
    full_join(rat_summ %>% select(Method, Cov_Rat = Cov, Size_Rat = Size, AUROC_Rat = AUROC), by = "Method") %>%
    arrange(factor(Method, levels = c("HDC", "Inv. quantile", "Penalized", 
                                      "Similarity", "CHDC-ratio", "CHDC-discount")))
  
  # LaTeX Formatting
  align_str <- "l|l|ccc|ccc|ccc|ccc|"
  additor <- list(pos = list(0), command = paste0(
    "\\hline\n",
    " & \\multicolumn{3}{c|}{\\textbf{MNIST ($\\alpha=", mnist_alpha, "$)}} ",
    "& \\multicolumn{3}{c|}{\\textbf{Languages ($\\alpha=", lang_alpha, "$)}} ",
    "& \\multicolumn{3}{c|}{\\textbf{ISOLET ($\\alpha=", iso_alpha, "$)}} ",
    "& \\multicolumn{3}{c|}{\\textbf{", rat_name, " ($\\alpha=", rat_alpha, "$)}} \\\\\n",
    "\\cline{2-13}\n",
    "\\textbf{Method} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} & \\textbf{Cov.} & \\textbf{Size} & \\textbf{AUC} \\\\\n",
    "\\hline\n"
  ))
  
  ltx <- xtable(final_tab, align = align_str)
  
  # Print/Save logic
  if (!is.null(save_path)) {
    if (!dir.exists(dirname(save_path))) dir.create(dirname(save_path), recursive = TRUE)
    print(ltx, file = save_path, floating = FALSE, include.rownames = FALSE, include.colnames = FALSE,
          sanitize.text.function = function(x) x, add.to.row = additor, hline.after = c(nrow(final_tab)), comment = FALSE)
    cat(paste("Saved combined tabular to:", save_path, "\n"))
  } else {
    print(ltx, floating = FALSE, include.rownames = FALSE, include.colnames = FALSE,
          sanitize.text.function = function(x) x, add.to.row = additor, hline.after = c(nrow(final_tab)), comment = FALSE)
  }
}

#------------------
# Execution
#------------------
table_dir <- '../../results/tables/benchmark_summary_with_rat.tex'

# Execute the function
create_combined_with_rat_table(
  iso_df      = isolet_data, 
  mnist_df    = mnist_data, 
  lang_df     = languages_data, 
  odor_df     = odor_data, 
  iso_alpha   = 0.02, 
  mnist_alpha = 0.05, 
  lang_alpha  = 0.01, 
  rat_name    = "Buchanan", 
  rat_alpha   = 0.2, 
  rat_beta    = 0.3, 
  save_path   = table_dir
)