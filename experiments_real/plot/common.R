
# --- A. Set-Valued Metrics (Marginal Only) ---
# spilt conformal
set_metrics <- dat %>%
  filter(method == "split_conformal") %>%
  summary_set_val()

# jk
set_metrics_jk <- dat %>%
  filter(method == "jackknife_plus") %>%
  summary_set_val()

# fcp
set_metrics_fcp <- dat %>%
  filter(method == "full_conformal") %>%
  summary_set_val()

# --- B. Point-Valued Metrics ---
point_metrics <- dat %>%
  summary_point_val()


# --- C. OOD Metrics ---
ood_metrics <- dat %>%
  summary_ood_val()


# --- D. Merge & Format ---
base <- tibble(score_type = unique(dat$score_type))

joined <- base %>%
  left_join(set_metrics     %>% rename_with(~ paste0("std_", .), -score_type), by = "score_type") %>%
  left_join(set_metrics_jk  %>% rename_with(~ paste0("jk_", .),  -score_type), by = "score_type") %>%
  left_join(set_metrics_fcp %>% rename_with(~ paste0("fcp_", .), -score_type), by = "score_type") %>%
  left_join(point_metrics, by = "score_type") %>%
  left_join(ood_metrics,   by = "score_type")

formatted <- joined %>%
  transmute(
    score_type = score_type,
    Cov_std  = fmt(std_m_cov_u, std_m_cov_se),
    Size_std = fmt(std_m_siz_u, std_m_siz_se),
    Cov_jk   = fmt(jk_m_cov_u, jk_m_cov_se),
    Size_jk  = fmt(jk_m_siz_u, jk_m_siz_se),
    Cov_fcp  = fmt(fcp_m_cov_u,   fcp_m_cov_se),
    Size_fcp = fmt(fcp_m_siz_u,   fcp_m_siz_se),
    Accuracy      = fmt(acc_u,     acc_se),
    AUROC      = fmt(ood_u,     ood_se)
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
                                    "Similarity", "CHDC-ratio", "CHDC-discount")))
formatted_standard = formatted%>%
  select(Method, Cov_std, Size_std, Accuracy, AUROC)
formatted_set_compare = formatted%>%
  select(Method, Cov_std, Size_std,Cov_jk, Size_jk,Cov_fcp, Size_fcp)


# --- F. Construct and save to LaTeX---
# Structure: Method | Cov Size | Acc | OOD |

export_latex_table(
  df = formatted_standard,
  align_str = "l|l|cc|c|c|",
  header_cmd = header_standard,
  save_dir = save_dir,
  file_name = if (exists("target_alpha")) paste0("isolet_table_alpha", target_alpha, ".tex") else "isolet_table.tex"
)

# Structure: Method | Cov Size | Cov Size | Cov Size
export_latex_table(
  df = formatted_set_compare,
  align_str = "l|l|cc|cc|cc|",
  header_cmd = header_set_compare,
  save_dir = save_dir,
  file_name = if (exists("target_alpha")) paste0("set_compare_table_alpha", target_alpha, ".tex") else "set_compare_table.tex"
)

#-------------------------------------------------------------------------------
#                           Common functions
#-------------------------------------------------------------------------------

summary_set_val<- function(data, ...){
  return(
    data %>%     
      filter(exp == "set_valued", marginal == TRUE) %>%
      group_by(score_type, ...) %>%
      summarise(
        n = n(),
        m_cov_u = mean(set_cov, na.rm=TRUE), 
        m_cov_se = sd(set_cov, na.rm=TRUE) / sqrt(n),
        m_siz_u = mean(set_size, na.rm=TRUE), 
        m_siz_se = sd(set_size, na.rm=TRUE) / sqrt(n),
        .groups = "drop"
      )
  )
}

summary_point_val <- function(data, ...){
  return(
    data %>% 
      filter(exp == "point_valued") %>%
      group_by(score_type, ...) %>%
      summarise(
        n = n(),
        acc_u = mean(point_acc, na.rm=TRUE), 
        acc_se = sd(point_acc, na.rm=TRUE) / sqrt(n),
        .groups = "drop"
      )
  )
}

summary_ood_val <- function(data, ...){
  return(
    data %>% 
      filter(exp == "ood", marginal == TRUE) %>%
      group_by(score_type, ...) %>%
      summarise(
        n = n(),
        ood_u = mean(ood_auroc, na.rm=TRUE), 
        ood_se = sd(ood_auroc, na.rm=TRUE) / sqrt(n),
        .groups = "drop"
      )
  )
}
#-------------------------------------------------------------------------------
#                           Formatting functions
#-------------------------------------------------------------------------------
 
# format_metric <- function(val_mean, val_se) {
#   ifelse(is.na(val_mean), "-", sprintf("%.3f (%.3f)", val_mean, val_se))
# }

# Vectorized helper function
fmt <- function(u, se) ifelse(is.na(u), "-", sprintf("%.3f (%.3f)", u, se))


header_standard <- paste0(
  "\\hline\n",
  " & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
  "\\cline{2-5}\n",
  "\\textbf{Method} & \\textbf{Coverage} & \\textbf{Size} & \\textbf{Accuracy} & \\textbf{AUC} \\\\\n",
  "\\hline\n"
)
header_set_compare <- paste0(
  "\\hline\n",
  " & \\multicolumn{2}{c|}{\\textbf{Standard}} & \\multicolumn{2}{c|}{\\textbf{Jackknife}} & \\multicolumn{2}{c|}{\\textbf{FCP}} \\\\\n",
  "\\cline{2-7}\n",
  "\\textbf{Method} & \\textbf{Coverage} & \\textbf{Size} & \\textbf{Coverage} & \\textbf{Size} & \\textbf{Coverage} & \\textbf{Size} \\\\\n",
  "\\hline\n"
)
export_latex_table <- function(df, align_str, header_cmd, 
                               group_col = NULL, 
                               save_dir = NULL, 
                               file_name = NULL) {
  
  # Automatically compute group lines if group_col exists in df; otherwise, just line at the end
  if (!is.null(group_col) && group_col %in% colnames(df)) {
    hlines <- cumsum(rle(as.character(df[[group_col]]))$lengths)
  } else {
    hlines <- c(nrow(df))
  }
  
  additor <- list(pos = list(0), command = header_cmd)
  ltx <- xtable::xtable(df, align = align_str)
  
  # Print / Save logic
  if (!is.null(save_dir) && !is.null(file_name)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filepath <- file.path(save_dir, file_name)
    
    xtable::print.xtable(
      ltx, file = filepath, floating = FALSE, 
      include.rownames = FALSE, include.colnames = FALSE, 
      sanitize.text.function = function(x) x, 
      add.to.row = additor, hline.after = hlines, comment = FALSE
    )
    cat("Saved tabular to:", filepath, "\n")
  } else {
    xtable::print.xtable(
      ltx, floating = FALSE, 
      include.rownames = FALSE, include.colnames = FALSE, 
      sanitize.text.function = function(x) x, 
      add.to.row = additor, hline.after = hlines, comment = FALSE
    )
  }
}
