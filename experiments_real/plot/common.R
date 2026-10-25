
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
library(ggplot2)
library(tidyr)
library(dplyr)

# RUNTIME_METHODS <- c(split_marginal = "Set-valued marg.", split_conditional = "Set-valued cond.",
#                      point_valued = "Point-valued", jackknife_plus = "Jackknife+",
#                      full_conformal = "Full conformal")
# 

# 
# # n_test: test points per repetition, stated in the header (times are not rescaled)
# create_runtime_table <- function(timing, target_alpha, score = "ratio") {
#   sec <- function(x) ifelse(is.na(x), "-", sprintf("%.4f", x))
#   tab <-   timing %>%
#     filter(abs(alpha - target_alpha) < 1e-6, score_type %in% c(score, "all")) %>%
#     group_by(method) %>%
#     summarise(across(c(training, calibration, encoding, hdc, conformal), mean), .groups = "drop") %>%
#     mutate(method = factor(method, levels = names(RUNTIME_METHODS))) %>%
#     arrange(method) %>%
#     mutate(method = RUNTIME_METHODS[as.character(method)]) %>%
#     mutate(across(-method, sec),
#            method = factor(method, levels = RUNTIME_METHODS))
#   return(tab)
# }
# 
# uci_har_timing = load_exp_results("./uci_har/", pattern = "^timing_seed.*_alpha.*\\.csv$")
# timing = create_runtime_table(uci_har_timing, target_alpha = 0.1)

plot_runtime <- function(uci_har_timing, target_alpha = 0.1, save_dir = ".") {
  
  lw <- 0.25 / (.pt * 0.75)
  
  stage_cols <- c(training    = "grey",
                  calibration = "#FD6467",   # coral
                  inference   = "lightgrey",
                  conformal   = "#B40F20")   # deep red
  
  long <- uci_har_timing %>%
    filter(abs(alpha - target_alpha) < 1e-6, score_type %in% c("ratio", "all")) %>%
    mutate(inference = encoding + hdc, .keep = "unused") %>%
    group_by(method) %>%
    summarise(across(names(stage_cols), \(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
    mutate(method = factor(RUNTIME_METHODS[method], levels = RUNTIME_METHODS)) %>%
    pivot_longer(-method, names_to = "stage", values_to = "time") %>%
    mutate(time  = replace_na(time, 0),
           stage = factor(stage, levels = names(stage_cols)))
  
  conf_lab <- long %>%
    group_by(method) %>%
    summarise(top  = sum(time),
              conf = time[stage == "conformal"]) %>%
    mutate(label = ifelse(conf < 1,
                          sprintf("%.1f ms", conf * 1000),
                          sprintf("%.2f s", conf)))
  
  cpu_name <- unique(uci_har_timing$cpu)[1]
  
  p_time <- ggplot(long, aes(x = method, y = time, fill = stage)) +
    geom_col(aes(linewidth = stage == "conformal",
                 color     = stage == "conformal"),
             width = 0.6, position = position_stack(reverse = TRUE)) +
    scale_fill_manual(values = stage_cols,
                      labels = c("Training", "Calibration", "Inference", "Conformal")) +
    scale_linewidth_manual(values = c(`FALSE` = lw, `TRUE` = 4 * lw), guide = "none") +
    scale_color_manual(values = c(`FALSE` = "grey30", `TRUE` = "#B40F20"), guide = "none") +
    geom_text(data = conf_lab, aes(x = method, y = top, label = label),
              inherit.aes = FALSE, vjust = -0.5, size = 2.2,
              color = "#B40F20", family = "Arial") +
    annotate("text", x = Inf, y = Inf, label = paste("CPU:", cpu_name),
             hjust = 1.02, vjust = 1.5, size = 2, color = "grey30", family = "Arial") +
    coord_cartesian(ylim = c(15, 30)) +
    labs(x = NULL, y = "Time (s)", fill = NULL) +
    theme_classic(base_size = 9, base_family = "Arial") +
    theme(axis.line   = element_line(linewidth = lw),
          axis.ticks  = element_line(linewidth = lw),
          axis.text.x = element_text(angle = 30, hjust = 1),
          legend.position = "top",
          legend.key.size = unit(3, "mm"))
  
  ggsave(file.path(save_dir, "timing.pdf"), p_time,
         width = 90, height = 80, units = "mm", device = cairo_pdf)
  
  p_time
}
