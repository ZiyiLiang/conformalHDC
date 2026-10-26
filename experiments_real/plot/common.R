
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
#                           plot
#-------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra) 
# RUNTIME table -------------------------------
lw <- 0.25 / (.pt * 0.75)
RUNTIME_METHODS <- c(split_marginal = "Set-valued marg.", split_conditional = "Set-valued cond.",
                     point_valued = "Point-valued", jackknife_plus = "Jackknife+",
                     full_conformal = "Full conformal")
# timing_seed*.csv rows: training, calibration (one-time), and per-repetition test
# encoding / HDC inference / conformal overhead on the full test split.
plot_runtime <- function(uci_har_timing, target_alpha = 0.1, save_dir = ".") {
  
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


# EXP performance-----------------------------------------

get_mean <- function(x) suppressWarnings(as.numeric(sub(" .*", "", trimws(x)))) 
get_se <- function(x) suppressWarnings(as.numeric(sub(".*\\((.*)\\).*", "\\1", trimws(x))))

score_colors <- c(
  "CHDC-ratio"    = "#972D15", 
  "CHDC-discount" = "#F4B5BD",   
  "HDC (train)"   = "#B5B2AE",   
  "HDC"           = "#8D8680",   
  "Inv. quantile" = "#6B7480",  
  "Penalized"     = "#7E8A84",   
  "Similarity"    = "#A3A1AD" )  


plot_exp_results <- function(results, target_alpha, delta_ylim = c(-0.01,0.01), save_path=NULL) {
  
  # ---- Data: split "mean (se)" cells into numeric columns ----
  dat <- results %>%
    ungroup() %>%
    mutate(method       = factor(Method, levels = unique(Method)),
           coverage     = get_mean(Cov_std),          coverage_se     = get_se(Cov_std),
           size         = get_mean(Size_std),         size_se         = get_se(Size_std),
           coverage_jk  = get_mean(Cov_jk),           coverage_jk_se  = get_se(Cov_jk),
           size_jk      = get_mean(Size_jk),          size_jk_se      = get_se(Size_jk),
           coverage_fcp = get_mean(Cov_fcp),          coverage_fcp_se = get_se(Cov_fcp),
           size_fcp     = get_mean(Size_fcp),         size_fcp_se     = get_se(Size_fcp),
           ap           = get_mean(AveragePrecision), ap_se           = get_se(AveragePrecision),
           auc          = get_mean(AUROC),            auc_se          = get_se(AUROC)) %>%
    filter(method != "HDC (train)")
  
  # ---- Shared bar plot: one bar per score, +/- se error bar ----
  bar_plot <- function(y, ylab, legend = FALSE) {
    se <- paste0(y, "_se")
    ggplot(dat, aes(x = method, y = .data[[y]], fill = method)) +
      geom_col(width = 0.5, alpha = 0.5) +
      geom_errorbar(aes(ymin = .data[[y]] - .data[[se]],
                        ymax = .data[[y]] + .data[[se]]), width = 0.05) +
      labs(y = ylab, x = "Methods") +
      scale_fill_manual(values = score_colors) +
      theme_classic() +
      theme(legend.position = if (legend) "right" else "none")
  } 
  p1 <- bar_plot("ap", "Average precision") +                    # 1. average precision
    coord_cartesian(ylim = c(0, 1))
  p2 <- bar_plot("coverage", "Coverage") +# 2. coverage (target 0.95)
    geom_hline(yintercept = 1-target_alpha, linetype = "dashed",alpha = 0.6) +
    coord_cartesian(ylim = c(1-target_alpha - 0.025, min(1 - target_alpha +0.025,1)))
  p3 <- bar_plot("size", "Set size")                             #    set size
  p4 <- bar_plot("auc", "OOD AUC", legend = TRUE)                # 3. OOD detection AUC
  
  # ---- 4. Retraining: difference of JK+ / full conformal from standard ----
  diff_long <- dat %>%
    select(method,
           coverage_std_mean = coverage,     coverage_std_se = coverage_se,
           coverage_jk_mean  = coverage_jk,  coverage_jk_se  = coverage_jk_se,
           coverage_fcp_mean = coverage_fcp, coverage_fcp_se = coverage_fcp_se,
           size_std_mean     = size,         size_std_se     = size_se,
           size_jk_mean      = size_jk,      size_jk_se      = size_jk_se,
           size_fcp_mean     = size_fcp,     size_fcp_se     = size_fcp_se) %>%
    pivot_longer(-method,
                 names_to      = c("metric", "variant", ".value"),
                 names_pattern = "(coverage|size)_(std|jk|fcp)_(mean|se)") %>%
    filter(!is.na(mean)) %>%
    mutate(variant = factor(variant, levels = c("std", "jk", "fcp"),
                            labels = c("Standard", "Jackknife+", "Full conformal")),
           metric  = factor(metric, levels = c("coverage", "size"),
                            labels = c("Coverage", "Set size"))) %>%
    group_by(metric, method) %>%
    mutate(diff = mean - mean[variant == "Standard"]) %>%
    ungroup() %>%
    filter(variant != "Standard")
  
  dodge <- position_dodge(width = 0.7)
  
  diff_plot <- function(m, ylab,ylim, legend = FALSE) {
    ggplot(filter(diff_long, metric == m),
           aes(x = method, y = diff, fill = method, alpha = variant)) +
      geom_hline(yintercept = 0, color = "grey40") +
      geom_col(position = dodge, width = 0.6) +
      geom_point(aes(color = method, shape = variant, group = variant),
                 position = dodge, size = 1.5, alpha = 1) +
      scale_color_manual(values = score_colors, guide = "none") +
      scale_fill_manual(values = score_colors, guide = "none") +
      scale_shape_manual(values = c(`Jackknife+` = 17, `Full conformal` = 16)) +
      scale_alpha_manual(values = c(`Jackknife+` = 0.3, `Full conformal` = 0.7)) +
      labs(x = "Scores", y = ylab, alpha = "Variant", shape = "Variant") +
      coord_cartesian(ylim =ylim) +
      theme_classic() +
      theme(legend.position      = if (legend) c(0.98, 0.98) else "none",
            legend.justification = c(1, 1),
            legend.key.size      = unit(3, "mm"),
            legend.background    = element_blank())
  }
  
  p_dcov  <- diff_plot("Coverage", "Δ Coverage", ylim = c(-0.01,0.01),legend = TRUE)
  p_dsize <- diff_plot("Set size", "Δ Set size",ylim = delta_ylim)
  # ---- Full figure: 7 pt Arial, 180 mm wide ----
  #   col 1: coverage, set size | col 2: AP, OOD AUC | col 3: retraining differences
  fig_theme <- theme(text        = element_text(family = "Arial", size = 7),
                     axis.text   = element_text(size = 7),
                     axis.text.x = element_text(angle = 30, hjust = 1),
                     strip.text  = element_text(size = 7),
                     legend.text = element_text(size = 7),
                     plot.title  = element_text(size = 9, face = "bold.italic"))
  f <- function(p) p + fig_theme
  
  no_x <- theme(axis.text.x  = element_blank(),
                axis.ticks.x = element_blank(),
                axis.title.x = element_blank())
  
  stack2 <- function(top, bottom) {
    rbind(ggplotGrob(f(top) + no_x), ggplotGrob(f(bottom)), size = "first")
  }
  
  blank <- ggtitle(" ")   # same height as a real title, keeps rows aligned
  
  fig <- arrangeGrob(
    stack2(p2 + ggtitle("Set-valued prediction"), p3 + blank),
    stack2(p1 + ggtitle("Point-valued prediction"),
           p4 + ggtitle("OOD detection") + theme(legend.position = "none")),
    stack2(p_dcov + ggtitle("Retrain variants"), p_dsize + blank),
    ncol = 3
  )
  ggsave(save_path, fig, width = 180, height = 120, units = "mm", device = cairo_pdf)
}


# Neural decoding -------------------
plot_odor_results = function(odor_table){
  
  nrats = unique(odor_table$rat) %>% length() 
  dat = odor_table %>%
    ungroup() %>%
    mutate(method      = factor(Method, levels = unique(Method)),
           coverage    = get_mean(Cov_std),  
           size        = get_mean(Size_std),
           ap          = get_mean(AveragePrecision),
           auc         = get_mean(AUROC)) %>% 
    filter(method != "HDC (train)")  %>% 
    select(method, rat, coverage, size, auc, ap)
  
  
  # ---- 1. Average precision ----
  p1 = ggplot(dat, aes(x = method, y = ap, fill = method)) +
    geom_boxplot(
      width = 0.3, outlier.shape = NA, linewidth = lw,
      alpha = 0.25, color = 'darkgrey') +
    geom_jitter(aes(shape = rat, color = method), width = 0.1, height = 0, size = 3) +
    labs(y = "Average precision", x = 'Methods') +
    scale_color_manual(values = score_colors) +
    scale_fill_manual(values = score_colors) + theme_classic() +theme(legend.position = 'none')
  
  
  # ---- 2. Coverage vs. set size (target coverage = 1 - alpha = 0.8) ----
  p2 = ggplot(dat, aes(x = method, y = coverage, fill = method)) +
    geom_boxplot(
      linewidth = lw, width = 0.3, outlier.shape = NA, alpha = 0.25, color = 'darkgrey') +
    geom_jitter(aes(shape = rat, color = method), width = 0.1, height = 0, size = 3) +
    labs(y = "Coverage", x = 'Methods') +
    geom_hline(yintercept = 0.8, linetype = "dashed") +
    ylim(0.75,0.85)+
    scale_color_manual(values = score_colors) +
    scale_fill_manual(values = score_colors)+theme_classic()+theme(legend.position = 'none')
  
  
  p3 = ggplot(dat, aes(x = method, y = size, fill = method)) +
    geom_boxplot(linewidth = lw,width = 0.3, outlier.shape = NA, alpha = 0.25, color = 'darkgrey') +
    geom_jitter(aes(shape = rat, color = method), width = 0.1, height = 0, size = 3) +
    labs(y = "Set size", x = 'Methods') + 
    scale_color_manual(values = score_colors) +
    scale_fill_manual(values = score_colors)+  theme_classic()+theme(legend.position = 'none')
  
  # ---- 3. OOD detection AUC (0.5 = chance) ---- 
  p4 = ggplot(dat, aes(x = method, y = auc, fill = method)) +
    geom_boxplot(linewidth = lw,width = 0.3, outlier.shape = NA, alpha = 0.3, color = 'darkgrey') +
    geom_jitter(aes(shape = rat, color = method), width = 0.1, height = 0, size = 3) + 
    labs(y = "OOD AUC", x = 'Methods') + 
    scale_color_manual(values = score_colors) +
    scale_fill_manual(values = score_colors)+ theme_classic()
  
  
  plots <- list(ap = p1, coverage = p2, size = p3, auc = p4)
  return(plots)
}
