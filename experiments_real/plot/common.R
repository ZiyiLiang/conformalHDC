
#-------------------------------------------------------------------------------
#                           load functions
#------------------------------------------------------------------------------- 
load_exp_results <- function(results_dir, pattern = "^seed.*_alpha.*\\.csv$"){ 
  
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
  "\\toprule\n",
  " & \\multicolumn{2}{@{}c@{}}{Set-Valued}",
  " & \\multicolumn{2}{@{}c@{}}{Point}",
  " & \\multicolumn{1}{@{}c@{}}{OOD} \\\\",
  "Method & Coverage & Size",
  " & Accuracy & Average Precision",
  " & AUC \\\\\n",
  "\\midrule\n"
)


header_set_compare <- paste0(
  "\\toprule\n",
  " & \\multicolumn{2}{@{}c@{}}{Standard} & \\multicolumn{2}{@{}c@{}}{Jackknife} & \\multicolumn{2}{@{}c@{}}{FCP} \\\\",
  "Method & Coverage & Size & Coverage & Size & Coverage & Size \\\\\n",
  "\\midrule\n"
)

export_latex_table <- function(df, align_str, header_cmd, 
                               group_col = NULL, 
                               save_dir = NULL, 
                               file_name = NULL) {
  
  df <- as.data.frame(df)  # drop tibble grouping
  
  # Automatically compute group lines if group_col exists in df; otherwise, just line at the end
  if (!is.null(group_col) && group_col %in% colnames(df)) {
    g <- trimws(as.character(df[[group_col]]))
    g[is.na(g)] <- ""
    if (any(g == "")) {
      # group label only on the first row of each group (e.g. \multirow), blanks below
      starts <- which(g != "")
      hlines <- c(starts[-1] - 1, nrow(df))
    } else {
      hlines <- cumsum(rle(g)$lengths)
    }
  } else {
    hlines <- c(nrow(df))
  }
  
  # Template style: no vertical rules, \midrule between groups, \botrule at the end
  align_str <- gsub("|", "", align_str, fixed = TRUE)
  n_rules <- length(hlines)
  additor <- list(
    pos = c(list(0), as.list(hlines)),
    command = c(header_cmd, rep("\\midrule\n", n_rules - 1), "\\botrule\n")
  )
  ltx <- xtable::xtable(df, align = paste0("r", align_str))
  
  tex <- xtable::print.xtable(
    ltx, floating = FALSE, print.results = FALSE,
    include.rownames = FALSE, include.colnames = FALSE, 
    sanitize.text.function = function(x) x, 
    add.to.row = additor, hline.after = NULL, comment = FALSE
  )
  tex <- sub(paste0("{", align_str, "}"), paste0("{@{}", align_str, "@{}}"), tex, fixed = TRUE)
  
  # Print / Save logic
  if (!is.null(save_dir) && !is.null(file_name)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filepath <- file.path(save_dir, file_name)
    cat(tex, file = filepath)
    cat("Saved tabular to:", filepath, "\n")
  } else {
    cat(tex)
  }
} 

#-------------------------------------------------------------------------------
#                           Runtime table
#-------------------------------------------------------------------------------
# timing_seed*.csv rows: training, calibration (one-time), and per-repetition test
# encoding / HDC inference / conformal overhead on the full test split.
RUNTIME_METHODS <- c(split_marginal = "Set-valued marg.", split_conditional = "Set-valued cond.",
                     point_valued = "Point-valued", jackknife_plus = "Jackknife+",
                     full_conformal = "Full conformal")

summary_runtime <- function(timing, target_alpha, score = "ratio") {
  timing %>%
    filter(abs(alpha - target_alpha) < 1e-6, score_type %in% c(score, "all")) %>%
    group_by(method) %>%
    summarise(across(c(training, calibration, encoding, hdc, conformal), mean), .groups = "drop") %>%
    mutate(method = factor(method, levels = names(RUNTIME_METHODS))) %>%
    arrange(method) %>%
    mutate(method = RUNTIME_METHODS[as.character(method)])
}

header_runtime <- function(n_test) paste0(
  "\\toprule\n",
  " & & & \\multicolumn{3}{@{}c@{}}{Inference (", n_test, " test points)} \\\\\n",
  "Method & Training (s) & Calibration (s) & Encoding (s) & HDC (s) & Conformal (s) \\\\\n",
  "\\midrule\n"
)

# Last line of the table: the CPU the times were measured on
hardware_line <- function(timing) sprintf(
  "\\multicolumn{6}{@{}l@{}}{CPU: %s} \\\\\n",
  paste(unique(timing$cpu), collapse = ", "))

# n_test: test points per repetition, stated in the header (times are not rescaled)
create_runtime_table <- function(timing, target_alpha, n_test, score = "ratio",
                                 save_dir = NULL, file_name = NULL) {
  sec <- function(x) ifelse(is.na(x), "-", sprintf("%.4f", x))
  tab <- summary_runtime(timing, target_alpha, score) %>%
    mutate(across(-method, sec))
  tex <- paste(capture.output(export_latex_table(tab, "lccccc", header_runtime(n_test))), collapse = "\n")
  tex <- sub("\\end{tabular}", paste0(hardware_line(timing), "\\end{tabular}"), tex, fixed = TRUE)
  if (is.null(save_dir) || is.null(file_name)) return(cat(tex, "\n"))
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
  cat(tex, "\n", file = file.path(save_dir, file_name))
  cat("Saved tabular to:", file.path(save_dir, file_name), "\n")
}
