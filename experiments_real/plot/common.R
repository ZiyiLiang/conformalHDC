
#-------------------------------------------------------------------------------
#                           load functions
#------------------------------------------------------------------------------- 
load_exp_results <- function(results_dir, pattern = "seed.*_alpha.*\\.csv"){ 
  
  # Pattern matches: seedX_alphaY.csv
  files <- list.files(path = results_dir, pattern = pattern, 
                      full.names = TRUE, recursive = TRUE)
  
  if (length(files) == 0) {
    warning("No result files found! Check your directory path.")
  } else {
    data <- files %>% 
      map_df(~read_csv(., show_col_types = FALSE))
    cat(paste("Loaded", length(files), "files. Total rows:", nrow(data), "\n"))
  }
  return(data)
}

#-------------------------------------------------------------------------------
#                           summary functions
#-------------------------------------------------------------------------------

# Treat HDC (vanilla_full) as producing a singleton set:
#   - Set Coverage = Point Accuracy
#   - Set Size     = 1.0
summary_vanilla_hdc <- function(data){

  return(
    data %>%
    filter(score_type == "vanilla_full", exp == "point_valued") %>%
    select(random_state, point_acc, macro_ap,macro_auprc, any_of(c("alpha", "sigma"))) %>%    
    mutate(
      exp = "set_valued",
      marginal = TRUE,
      score_type = "vanilla_full",
      set_cov = point_acc, # Coverage becomes Accuracy
      set_size = 1.0       # Size is always 1
    ))
  
}

# Conformal HDC
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
        ap_u = mean(macro_ap, na.rm=TRUE), 
        ap_se = sd(macro_ap, na.rm=TRUE) / sqrt(n), 
        auprc_u = mean(macro_auprc, na.rm=TRUE), 
        auprc_se = sd(macro_auprc, na.rm=TRUE) / sqrt(n),
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
  " & \\multicolumn{2}{c|}{\\textbf{Set-Valued}}",
  " & \\multicolumn{2}{c|}{\\textbf{Point}}",         
  " & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",  
  "\\textbf{Method} & \\textbf{Coverage} & \\textbf{Size}",
  " & \\textbf{Accuracy} & \\textbf{Average Precision}",  
  " & \\textbf{AUC} \\\\\n",
  "\\hline\n"
)


header_set_compare <- paste0(
  "\\hline\n",
  " & \\multicolumn{2}{c|}{\\textbf{Standard}} & \\multicolumn{2}{c|}{\\textbf{Jackknife}} & \\multicolumn{2}{c|}{\\textbf{FCP}} \\\\\n", 
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
  ltx <- xtable::xtable(df, align = paste0("r", align_str))
  
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
