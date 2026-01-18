# Install packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("xtable")) install.packages("xtable")

library(tidyverse)
library(xtable)

#-------------------------------------------------------------------------------
#                            Odor Decoding Exp
#-------------------------------------------------------------------------------
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_real/results/")
results_dir <- "./odor_decoding/" 

# Pattern matches: ratX_seedY_alphaZ_betaW.csv
files <- list.files(path = results_dir, pattern = "rat.*_seed.*_alpha.*_beta.*\\.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found! Check your directory path.")
}

full_data <- files %>% 
  map_df(~read_csv(., show_col_types = FALSE))

cat(paste("Loaded", length(files), "files. Total rows:", nrow(full_data), "\n"))

#------------------
# Table Generation 
#------------------
format_metric <- function(val_mean, val_se) {
  # If value is NA return empty string
  if (is.na(val_mean)) return("-")
  
  # Format string: Mean (SE)
  txt <- sprintf("%.3f (%.3f)", val_mean, val_se)
  return(txt)
}

create_rat_table <- function(df, target_rat, target_alpha, target_beta,
                             save_dir = NULL, plot_conditional = FALSE) {
  
  # Filter Data: Added Beta Filter
  dat <- df %>% 
    filter(rat_id == target_rat, 
           abs(alpha - target_alpha) < 1e-6 | is.na(alpha),
           abs(beta - target_beta) < 1e-6 | is.na(beta)) # Handle NA for vanilla if needed
  
  if (nrow(dat) == 0) return(NULL)
  
  # --- A. Set-Valued Metrics ---
  # 1. Marginal Sets
  set_marg <- dat %>%
    filter(exp == "set_valued", marginal == TRUE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      m_cov_u = mean(set_cov, na.rm=TRUE), 
      m_cov_se = sd(set_cov, na.rm=TRUE) / sqrt(n),
      m_siz_u = mean(set_size, na.rm=TRUE), 
      m_siz_se = sd(set_size, na.rm=TRUE) / sqrt(n)
    )
  
  # 2. Conditional Sets
  set_cond <- dat %>%
    filter(exp == "set_valued", marginal == FALSE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      c_mincov_u = mean(min_class_cov, na.rm=TRUE), 
      c_mincov_se = sd(min_class_cov, na.rm=TRUE) / sqrt(n),
      c_siz_u = mean(set_size, na.rm=TRUE), 
      c_siz_se = sd(set_size, na.rm=TRUE) / sqrt(n)
    )
  
  # --- B. Point-Valued Metrics ---
  # Modified: Removed "accurate", kept "efficient" and "vanilla"
  point_metrics <- dat %>%
    filter(exp == "point_valued", method %in% c("efficient", "vanilla")) %>%
    group_by(score_type, method) %>%
    summarise(
      n = n(),
      acc_u = mean(point_acc, na.rm=TRUE), 
      acc_se = sd(point_acc, na.rm=TRUE) / sqrt(n), 
      .groups = 'drop'
    ) %>%
    pivot_wider(names_from = method, values_from = c(acc_u, acc_se))
  
  # --- C. OOD Metrics ---
  ood_metrics <- dat %>%
    filter(exp == "ood", marginal == TRUE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      ood_u = mean(ood_auroc, na.rm=TRUE), 
      ood_se = sd(ood_auroc, na.rm=TRUE) / sqrt(n)
    )
  
  # --- D. Merge & Formatting ---
  all_scores <- unique(dat$score_type)
  base <- tibble(score_type = all_scores)
  
  joined <- base %>%
    left_join(set_marg, by="score_type") %>%
    left_join(set_cond, by="score_type") %>% 
    left_join(point_metrics, by="score_type") %>%
    left_join(ood_metrics, by="score_type")
  
  formatted <- joined %>%
    rowwise() %>%
    mutate(
      # Marginal Sets
      `Marg Cov` = format_metric(m_cov_u, m_cov_se),
      `Marg Size` = format_metric(m_siz_u, m_siz_se),
      
      # Conditional Sets
      `Cond MinCov` = format_metric(c_mincov_u, c_mincov_se),
      `Cond Size` = format_metric(c_siz_u, c_siz_se),
      
      # Point Acc (Renamed to generic Accuracy)
      # We use 'efficient' as the primary source for conformal methods
      `Accuracy` = if("acc_u_efficient" %in% names(.)) format_metric(acc_u_efficient, acc_se_efficient) else "-",
      
      # Handle Vanilla specifically: Vanilla usually doesn't have 'efficient' method name
      # It maps to method='vanilla', so we check acc_u_vanilla
      `Accuracy` = if(score_type == "cosine" && "acc_u_vanilla" %in% names(.)) 
        format_metric(acc_u_vanilla, acc_se_vanilla) else `Accuracy`,
      
      # OOD
      `AUROC` = format_metric(ood_u, ood_se)
    ) %>%
    ungroup()
  
  # --- E. Rename and Reorder Methods ---
  formatted <- formatted %>%
    mutate(Method = case_when(
      score_type == "cosine" ~ "HDC",
      score_type == "inverse_quantile" ~ "Inv-Quantile",
      score_type == "penalized" ~ "Penalized",
      score_type == "sim" ~ "Sim",
      score_type == "ratio" ~ "CHDC-ratio",
      score_type == "discount" ~ "CHDC-discount",
      TRUE ~ score_type
    )) %>%
    arrange(factor(Method, levels = c("HDC", "Inv-Quantile", "Penalized", "Sim", "CHDC-ratio", "CHDC-discount")))
  
  # --- F. Column Selection & Header Construction ---
  additor <- list()
  additor$pos <- list(0)
  
  if (plot_conditional) {
    # LONG TABLE
    final_df <- formatted %>% select(Method, `Marg Cov`, `Marg Size`, `Cond MinCov`, `Cond Size`, `Accuracy`, AUROC)
    align_str <- "ll|cccc|c|c|"
    
    additor$command <- paste0(
      "\\hline\n",
      " & \\multicolumn{4}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point-Valued}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
      "\\cline{2-7}\n",
      "\\textbf{Method} & \\textbf{Marg Cov} & \\textbf{Marg Size} & \\textbf{Cond MinCov} & \\textbf{Cond Size} & \\textbf{Accuracy} & \\textbf{AUROC} \\\\\n",
      "\\hline\n"
    )
    
  } else {
    # SHORT TABLE
    final_df <- formatted %>% 
      select(Method, `Marg Cov`, `Marg Size`, `Accuracy`, AUROC) %>%
      rename(Cov = `Marg Cov`, Size = `Marg Size`)
    
    align_str <- "ll|cc|c|c|"
    
    additor$command <- paste0(
      "\\hline\n",
      " & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point-Valued}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
      "\\cline{2-5}\n",
      "\\textbf{Method} & \\textbf{Cov} & \\textbf{Size} & \\textbf{Accuracy} & \\textbf{AUROC} \\\\\n",
      "\\hline\n"
    )
  }
  
  # --- G. Output ---
  ltx <- xtable(final_df, 
                caption = paste0("Results for Rat ", target_rat, " ($\\alpha=", target_alpha, ", \\beta=", target_beta, "$). Values are Mean (SE)."),
                label = paste0("tab:rat", target_rat, "_alpha", target_alpha, "_beta", target_beta),
                align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    fname_suffix <- if(plot_conditional) "_cond" else ""
    filename <- file.path(save_dir, paste0("table_rat", target_rat, "_alpha", target_alpha, "_beta", target_beta, fname_suffix, ".tex"))
    
    print(ltx, file = filename, include.rownames = FALSE, 
          include.colnames = FALSE, 
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = c(nrow(final_df)),
          comment = FALSE, caption.placement = "top")
    
    cat(paste("Saved:", filename, "\n"))
  } else {
    print(ltx, include.rownames = FALSE, include.colnames = FALSE,
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = c(nrow(final_df)),
          comment = FALSE)
  }
}

# -------------
#  Save tables
# -------------
rats_to_process <- 0:4
alphas_to_process <- c(0.2)
betas_to_process <- c(0.3)

table_dir <- '../../results/tables/odor_decoding/'

for (r in rats_to_process) {
  for (a in alphas_to_process) {
    for (b in betas_to_process) {
      # Short table
      create_rat_table(full_data, r, a, b, save_dir=table_dir, plot_conditional = FALSE)
      # # Long table
      # create_rat_table(full_data, r, a, b, save_dir=table_dir, plot_conditional = TRUE)
    }
  }
}