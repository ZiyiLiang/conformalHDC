# Install packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("cowplot")) install.packages("cowplot") 
if (!require("xtable")) install.packages("xtable")

library(grid) 
library(tidyverse)
library(gridExtra)
library(cowplot)
library(xtable)

#-------------------------------------------------------------------------------
#                           3-Class Exp Plots
#-------------------------------------------------------------------------------
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_synthetic/results/") 
results_dir <- "./3class/" 

files <- list.files(path = results_dir, pattern = "seed.*_alpha.*\\.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found!")
} else {
  full_data <- files %>% map_df(~read_csv(., show_col_types = FALSE))
}

#------------------
# Plotting Function
#------------------
create_3class_plot <- function(df, target_alpha, save_dir = NULL) {
  
  # Filter Data by Alpha
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha))
  
  if (nrow(dat) == 0) return(NULL)
  
  # --- 1. Compute Summaries ---
  acc_df <- dat %>% filter(exp == "point_valued") %>% group_by(sigma, score_type) %>%
    summarise(mean = mean(point_acc, na.rm=TRUE), se = sd(point_acc, na.rm=TRUE)/sqrt(n()), .groups = "drop") %>% mutate(Metric = "Accuracy")
  
  ood_df <- dat %>% filter(exp == "ood", marginal == TRUE) %>% group_by(sigma, score_type) %>%
    summarise(mean = mean(ood_auroc, na.rm=TRUE), se = sd(ood_auroc, na.rm=TRUE)/sqrt(n()), .groups = "drop") %>% mutate(Metric = "AUROC")
  
  cov_df <- dat %>% filter(exp == "set_valued", marginal == TRUE) %>% group_by(sigma, score_type) %>%
    summarise(mean = mean(set_cov, na.rm=TRUE), se = sd(set_cov, na.rm=TRUE)/sqrt(n()), .groups = "drop") %>% mutate(Metric = "Coverage")
  
  siz_df <- dat %>% filter(exp == "set_valued", marginal == TRUE) %>% group_by(sigma, score_type) %>%
    summarise(mean = mean(set_size, na.rm=TRUE), se = sd(set_size, na.rm=TRUE)/sqrt(n()), .groups = "drop") %>% mutate(Metric = "Size")
  
  # Combine
  plot_data <- bind_rows(acc_df, ood_df, cov_df, siz_df) %>%
    mutate(Method = case_when(
      score_type == "vanilla_full" ~ "HDC",
      score_type == "inverse_quantile" ~ "Inv. quantile",
      score_type == "penalized" ~ "Penalized",
      score_type == "sim" ~ "Similarity",
      score_type == "ratio" ~ "CHDC-ratio",
      score_type == "discount" ~ "CHDC-discount",
      TRUE ~ score_type
    )) %>%
    filter(Method != "vanilla_train")
  
  # Define Factor Levels
  method_levels <- c("HDC", "Inv. quantile", "Penalized", "Similarity", "CHDC-ratio", "CHDC-discount")
  plot_data$Method <- factor(plot_data$Method, levels = method_levels)
  
  # --- 2. Custom Colors ---
  custom_colors <- c(
    "HDC"             = "#000000",  
    "Inv. quantile"   = "#7bd5e4",  
    "Penalized"       = "#4575a0",  
    "Similarity"      = "#a1d991",  
    "CHDC-ratio"      = "#FF9900", 
    "CHDC-discount"   = "#CC3333"
  )
  
  # Common Theme (Larger Fonts)
  common_theme <- theme_bw() +
    theme(
      legend.position = "none", 
      axis.title = element_text(size = 18),
      axis.text = element_text(size = 14),
      strip.text = element_text(size = 16), 
      strip.background = element_rect(fill = "grey90")   
    )
  
  # --- 3. Create Plots ---
  # We use facet_wrap(~Metric) to get the grey banner header back
  
  p1 <- ggplot(plot_data %>% filter(Metric == "Accuracy"), 
               aes(x = sigma, y = mean, color = Method, fill = Method)) +
    geom_line(linewidth = 1.2) + 
    geom_ribbon(aes(ymin = mean - se, ymax = mean + se), alpha = 0.4, color = NA) +
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    facet_wrap(~Metric) + 
    labs(x = expression(sigma), y = NULL) +
    common_theme
  
  p2 <- ggplot(plot_data %>% filter(Metric == "AUROC", Method != "Inv. quantile"), 
               aes(x = sigma, y = mean, color = Method, fill = Method)) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(aes(ymin = mean - se, ymax = mean + se), alpha = 0.4, color = NA) +
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    facet_wrap(~Metric) + 
    labs(x = expression(sigma), y = NULL) +
    common_theme
  
  p3 <- ggplot(plot_data %>% filter(Metric == "Coverage"), 
               aes(x = sigma, y = mean, color = Method, fill = Method)) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(aes(ymin = mean - se, ymax = mean + se), alpha = 0.4, color = NA) +
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    coord_cartesian(ylim = c(0.86, 0.94)) + 
    geom_hline(yintercept = 1-target_alpha, linetype="dashed", color="gray40", linewidth=0.8) +
    facet_wrap(~Metric) + 
    labs(x = expression(sigma), y = NULL) +
    common_theme
  
  p4 <- ggplot(plot_data %>% filter(Metric == "Size"), 
               aes(x = sigma, y = mean, color = Method, fill = Method)) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(aes(ymin = mean - se, ymax = mean + se), alpha = 0.4, color = NA) +
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    facet_wrap(~Metric) + 
    labs(x = expression(sigma), y = NULL) +
    common_theme
  
  # --- 4. Shared Legend Extraction ---
  legend_plot <- ggplot(plot_data, aes(x = sigma, y = mean, color = Method)) +
    geom_line(linewidth = 2) + 
    scale_color_manual(values = custom_colors) +
    theme_bw() +
    theme(
      legend.position = "bottom", 
      legend.title = element_blank(), 
      legend.text = element_text(size=15), 
      legend.key.width = unit(1.5, "cm") 
    ) +
    guides(color = guide_legend(nrow = 1)) 
  
  shared_legend <- get_legend(legend_plot)
  
  # --- 5. Assembly ---
  plots_row <- arrangeGrob(p1, p2, p3, p4, nrow = 1)
  
  final_object <- arrangeGrob(
    plots_row,
    shared_legend,
    nrow = 2,
    heights = c(1, 0.15)
  )
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filename <- file.path(save_dir, paste0("3class_plot_alpha", target_alpha, ".pdf"))
    
    ggsave(filename, final_object, width = 12, height = 3.5) 
    cat(paste("Saved plot:", filename, "\n"))
  } else {
    grid.draw(final_object)
  }
}


#------------------
# Table generation
#------------------
format_metric <- function(val_mean, val_se) {
  if (is.na(val_mean)) return("-")
  # Use 3 decimal places as per your example
  sprintf("%.3f (%.3f)", val_mean, val_se)
}

create_3class_table <- function(df, target_alpha, save_dir = NULL) {
  
  # 1. Filter Data by Alpha
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha)) %>%
    filter(score_type != "vanilla_train")
  
  if (nrow(dat) == 0) {
    warning("No data found for the specified alpha.")
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
    select(random_state, sigma, point_acc) %>%   
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
  # Aggregation (Grouped by Sigma and Score Type)
  #------------------------------------------------
  
  # --- A. Set-Valued Metrics (Marginal Only) ---
  set_marg <- dat %>%
    filter(exp == "set_valued", marginal == TRUE) %>%
    group_by(sigma, score_type) %>%
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
    group_by(sigma, score_type) %>%
    summarise(
      n = n(),
      acc_u = mean(point_acc, na.rm=TRUE), 
      acc_se = sd(point_acc, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- C. OOD Metrics ---
  ood_metrics <- dat %>%
    filter(exp == "ood", marginal == TRUE) %>%
    group_by(sigma, score_type) %>%
    summarise(
      n = n(),
      ood_u = mean(ood_auroc, na.rm=TRUE), 
      ood_se = sd(ood_auroc, na.rm=TRUE) / sqrt(n),
      .groups = "drop"
    )
  
  # --- D. Merge All Metrics ---
  base <- dat %>% select(sigma, score_type) %>% distinct()
  
  joined <- base %>%
    left_join(set_marg, by=c("sigma", "score_type")) %>%
    left_join(point_metrics, by=c("sigma", "score_type")) %>%
    left_join(ood_metrics, by=c("sigma", "score_type"))
  
  # Format numeric values
  formatted <- joined %>%
    mutate(
      `Marg Cov` = mapply(format_metric, m_cov_u, m_cov_se),
      `Marg Size` = mapply(format_metric, m_siz_u, m_siz_se),
      `Accuracy` = mapply(format_metric, acc_u, acc_se),
      `AUROC` = mapply(format_metric, ood_u, ood_se)
    )
  
  # --- E. Rename & Reorder Methods ---
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
    # Define method order
    mutate(Method = factor(Method, levels = c("HDC", "Inv. quantile", "Penalized", 
                                              "Similarity", "CHDC-ratio", "CHDC-discount"))) %>%
    # Sort: First by Sigma, then by Method
    arrange(sigma, Method)
  
  #---------------------------------
  # Stitching - Final Table Layout
  #---------------------------------
  final_df <- formatted %>%
    group_by(sigma) %>%
    # Use \multirow for vertical centering. 
    # Creates "\multirow{N}{*}{Value}" for the 1st row, empty string for others.
    mutate(Sigma_Display = if_else(
      row_number() == 1, 
      paste0("\\multirow{", n(), "}{*}{", sigma, "}"), 
      ""
    )) %>%
    ungroup() %>%
    select(Sigma_Display, Method, `Marg Cov`, `Marg Size`, Accuracy, AUROC) %>%
    rename(Sigma = Sigma_Display, Cov = `Marg Cov`, Size = `Marg Size`)
  
  # --- F. LaTeX Header Construction ---
  # Alignment: 7 chars for 6 columns + 1 (hidden) rowname column
  # Structure: Rowname | Sigma Method | Cov Size | Acc | OOD |
  align_str <- "l|ll|cc|c|c|"
  
  additor <- list()
  additor$pos <- list(0)
  additor$command <- paste0(
    "\\hline\n",
    " & & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
    "\\cline{3-6}\n",
    "\\textbf{Sigma} & \\textbf{Method} & \\textbf{Coverage} & \\textbf{Size} & \\textbf{Accuracy} & \\textbf{AUC} \\\\\n",
    "\\hline\n"
  )
  
  # --- G. Horizontal Lines Between Sigma Groups ---
  sigma_rle <- rle(as.character(formatted$sigma))
  group_ends <- cumsum(sigma_rle$lengths)
  hlines <- group_ends
  
  # --- H. Save Table ---
  ltx <- xtable(final_df, align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filename <- file.path(save_dir, paste0("3class_table_alpha", target_alpha, ".tex"))
    
    print(ltx, file = filename, 
          floating = FALSE,         # <--- Creates just the tabular environment
          include.rownames = FALSE, 
          include.colnames = FALSE, 
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = hlines,
          comment = FALSE, 
          caption.placement = "top")
    
    cat(paste("Saved clean tabular to:", filename, "\n"))
  } else {
    print(ltx, 
          floating = FALSE,
          include.rownames = FALSE, 
          include.colnames = FALSE,
          sanitize.text.function = function(x){x}, 
          add.to.row = additor, 
          hline.after = hlines,
          comment = FALSE)
  }
}

# plot_dir <- '../../results/plots/'
# create_3class_plot(full_data, target_alpha = 0.1,  save_dir = plot_dir)

table_dir <- '../../results/tables/'
create_3class_table(full_data, target_alpha = 0.1, save_dir = table_dir)