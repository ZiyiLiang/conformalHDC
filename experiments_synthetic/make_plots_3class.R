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
#                            3-Class Exp Plots
#-------------------------------------------------------------------------------
# Set your results directory here
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_synthetic/results/")
results_dir <- "./3class/" 

# Load all 3-class files (ensure they contain the 'sigma' column)
files <- list.files(path = results_dir, pattern = "seed.*_alpha.*\\.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found!")
} else {
  full_data <- files %>% 
    map_df(~read_csv(., show_col_types = FALSE))
  cat(paste("Loaded", length(files), "files.\n"))
}

#------------------
# Plotting Function
#------------------
create_3class_plot <- function(df, target_alpha, save_dir = NULL) {
  
  # Filter Data by Alpha (keep NA alphas for Point Valued/Baselines)
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha))
  
  if (nrow(dat) == 0) return(NULL)
  
  # --- 1. Compute Summaries for Each Metric ---
  
  # A. Accuracy (Point Valued)
  acc_df <- dat %>%
    filter(exp == "point_valued") %>%
    group_by(sigma, score_type) %>%
    summarise(
      mean = mean(point_acc, na.rm=TRUE), 
      se = sd(point_acc, na.rm=TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(Metric = "Accuracy")
  
  # B. AUROC (OOD, Marginal)
  ood_df <- dat %>%
    filter(exp == "ood", marginal == TRUE) %>%
    group_by(sigma, score_type) %>%
    summarise(
      mean = mean(ood_auroc, na.rm=TRUE), 
      se = sd(ood_auroc, na.rm=TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(Metric = "AUROC")
  
  # C. Coverage (Set Valued, Marginal)
  cov_df <- dat %>%
    filter(exp == "set_valued", marginal == TRUE) %>%
    group_by(sigma, score_type) %>%
    summarise(
      mean = mean(set_cov, na.rm=TRUE), 
      se = sd(set_cov, na.rm=TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(Metric = "Coverage")
  
  # D. Size (Set Valued, Marginal)
  siz_df <- dat %>%
    filter(exp == "set_valued", marginal == TRUE) %>%
    group_by(sigma, score_type) %>%
    summarise(
      mean = mean(set_size, na.rm=TRUE), 
      se = sd(set_size, na.rm=TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(Metric = "Size")
  
  # --- 2. Combine & Format ---
  plot_data <- bind_rows(acc_df, ood_df, cov_df, siz_df) %>%
    mutate(Method = case_when(
      score_type == "vanilla_full" ~ "HDC",
      score_type == "sim" ~ "Sim",
      score_type == "ratio" ~ "Ratio",
      score_type == "discount" ~ "Discount",
      score_type == "penalized" ~ "Penalized",
      score_type == "inverse_quantile" ~ "Inv-Quantile",
      TRUE ~ score_type
    )) %>%
    filter(Method != "vanilla_train") %>% # Exclude Vanilla (Train)
    mutate(
      Method = factor(Method, levels = c("HDC",  "Inv-Quantile", "Penalized", "Sim", "Ratio", "Discount")),
      Metric = factor(Metric, levels = c("Accuracy", "AUROC", "Coverage", "Size"))
    )
  
  # --- 3. Generate Plot (1 row x 4 cols) ---
  p <- ggplot(plot_data, aes(x = sigma, y = mean, color = Method, fill = Method)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean - se, ymax = mean + se), alpha = 0.2, color = NA) +
    facet_wrap(~Metric, nrow = 1, scales = "free_y") +
    labs(x = expression(sigma), y = "Value", title = paste("3-Class Experiment ( alpha =", target_alpha, ")")) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold", size = 12),
      axis.title = element_text(size = 11)
    )
  
  # --- 4. Save/Print ---
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    filename <- file.path(save_dir, paste0("3class_plot_alpha", target_alpha, ".pdf"))
    
    # Save as wide PDF (e.g., 12x4 inches)
    ggsave(filename, p, width = 12, height = 4)
    cat(paste("Saved plot:", filename, "\n"))
  } else {
    print(p)
  }
}

plot_dir <- '../../results/plots/3class/'
create_3class_plot(full_data, target_alpha = 0.1,  save_dir = plot_dir)