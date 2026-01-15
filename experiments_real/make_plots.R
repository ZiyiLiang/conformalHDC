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

# Find all CSV files matching the pattern "rat*_seed*_alpha*.csv"
files <- list.files(path = results_dir, pattern = "rat.*_seed.*_alpha.*\\.csv", 
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
format_metric <- function(val_mean, val_se, is_cov = FALSE, alpha = 0.1) {
  # If value is NA return empty string
  if (is.na(val_mean)) return("-")
  
  # Format string: Mean (SE)
  txt <- sprintf("%.3f (%.3f)", val_mean, val_se)
  
  # Apply Red Highlight for Coverage violation
  # Condition: Mean < (Target - SE)
  if (is_cov) {
    target <- 1 - alpha
    lower_bound <- target - val_se
    
    if (val_mean < lower_bound) {
      return(paste0("\\textcolor{red}{", txt, "}"))
    }
  }
  return(txt)
}

create_rat_table <- function(df, target_rat, target_alpha, 
                             save_dir = NULL, plot_conditional = FALSE) {
  
  # Filter Data
  dat <- df %>% 
    filter(rat_id == target_rat, abs(alpha - target_alpha) < 1e-6 | is.na(alpha))
  
  if (nrow(dat) == 0) return(NULL)
  
  # --- A. Set-Valued Metrics (Calculate SE) ---
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
  
  # --- B. Point-Valued Metrics (Calculate SE) ---
  point_metrics <- dat %>%
    filter(exp == "point_valued", method %in% c("efficient", "accurate", "vanilla")) %>%
    group_by(score_type, method) %>%
    summarise(
      n = n(),
      acc_u = mean(point_acc, na.rm=TRUE), 
      acc_se = sd(point_acc, na.rm=TRUE) / sqrt(n), 
      .groups = 'drop'
    ) %>%
    pivot_wider(names_from = method, values_from = c(acc_u, acc_se))
  
  # --- C. OOD Metrics (Calculate SE) ---
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
      `Marg Cov` = format_metric(m_cov_u, m_cov_se, is_cov=TRUE, alpha=target_alpha),
      `Marg Size` = format_metric(m_siz_u, m_siz_se),
      
      # Conditional Sets
      `Cond MinCov` = format_metric(c_mincov_u, c_mincov_se, is_cov=TRUE, alpha=target_alpha),
      `Cond Size` = format_metric(c_siz_u, c_siz_se),
      
      # Point Acc (Robust checks for columns)
      `Acc (Eff)` = if("acc_u_efficient" %in% names(.)) format_metric(acc_u_efficient, acc_se_efficient) else "-",
      `Acc (Acc)` = if("acc_u_accurate" %in% names(.)) format_metric(acc_u_accurate, acc_se_accurate) else "-",
      
      # Handle Vanilla specifically
      `Acc (Eff)` = if(score_type=="cosine" && "acc_u_vanilla" %in% names(.)) 
        format_metric(acc_u_vanilla, acc_se_vanilla) else `Acc (Eff)`,
      
      # OOD
      `AUROC` = format_metric(ood_u, ood_se)
    ) %>%
    ungroup()
  
  # Clean Method Names and Order
  formatted <- formatted %>%
    mutate(Method = case_when(
      score_type == "cosine" ~ "Vanilla HDC",
      score_type == "sim" ~ "Conformal (Sim)",
      score_type == "ratio" ~ "Conformal (Ratio)",
      score_type == "discount" ~ "Conformal (Discount)",
      score_type == "penalized" ~ "Conformal (Penalized)",
      score_type == "inverse_quantile" ~ "Conformal (Inv-Quantile)",
      TRUE ~ score_type
    )) %>%
    arrange(factor(Method, levels = c("Vanilla HDC", "Conformal (Sim)", "Conformal (Ratio)", 
                                      "Conformal (Discount)", "Conformal (Penalized)", "Conformal (Inv-Quantile)")))
  
  # --- E. Column Selection based on Argument ---
  if (plot_conditional) {
    final_df <- formatted %>% select(Method, `Marg Cov`, `Marg Size`, `Cond MinCov`, `Cond Size`, `Acc (Eff)`, `Acc (Acc)`, AUROC)
    align_str <- "l|l|cc|cc|cc|c|"
    # Header Construction
    additor <- list()
    additor$pos <- list(0)
    additor$command <- paste0(
      "\\hline\n",
      " & \\multicolumn{2}{c|}{\\textbf{Marginal Sets}} & \\multicolumn{2}{c|}{\\textbf{Conditional Sets}} & \\multicolumn{2}{c|}{\\textbf{Point Accuracy}} & \\textbf{OOD} \\\\\n",
      "\\cline{2-8}\n" 
    )
  } else {
    final_df <- formatted %>% select(Method, `Marg Cov`, `Marg Size`, `Acc (Eff)`, `Acc (Acc)`, AUROC)
    align_str <- "l|l|cc|cc|c|"
    # Header Construction
    additor <- list()
    additor$pos <- list(0)
    additor$command <- paste0(
      "\\hline\n",
      " & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{2}{c|}{\\textbf{Point Accuracy}} & \\textbf{OOD} \\\\\n",
      "\\cline{2-6}\n"
    )
  }
  
  # --- F. Output ---
  ltx <- xtable(final_df, 
                caption = paste0("Results for Rat ", target_rat, " ($\\alpha=", target_alpha, "$). Values are Mean (SE). Red indicates coverage $< 1-\\alpha-SE$."),
                label = paste0("tab:rat", target_rat, "_alpha", target_alpha),
                align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    fname_suffix <- if(plot_conditional) "_cond" else ""
    filename <- file.path(save_dir, paste0("table_rat", target_rat, "_alpha", target_alpha, fname_suffix, ".tex"))
    
    print(ltx, file = filename, include.rownames = FALSE, 
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, include.colnames = TRUE,
          comment = FALSE, caption.placement = "top")
    cat(paste("Saved:", filename, "\n"))
  } else {
    print(ltx, include.rownames = FALSE, sanitize.text.function = function(x){x}, 
          add.to.row = additor, comment = FALSE)
  }
}

# -------------
#  Save tables
# -------------
rats_to_process <- 0:4
alphas_to_process <- c(0.1, 0.2, 0.3)
table_dir <- '../../results/tables/odor_decoding/'
for (r in rats_to_process) {
  for (a in alphas_to_process) {
    # Short table
    create_rat_table(full_data, r, a, save_dir=table_dir, plot_conditional = FALSE)
    # Long table
    create_rat_table(full_data, r, a, save_dir=table_dir, plot_conditional = TRUE)
  }
}