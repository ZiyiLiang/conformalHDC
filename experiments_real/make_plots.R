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


#-------------------------------------------------------------------------------
#                                MNIST Exp 
#-------------------------------------------------------------------------------
# Set your results directory here
setwd("C:/Users/liang/Documents/GitHub/conformalHDC/experiments_real/results/")
results_dir <- "./mnist/" 

# Pattern matches: seedX_alphaY.csv or similar
files <- list.files(path = results_dir, pattern = "*.csv", 
                    full.names = TRUE, recursive = TRUE)

if (length(files) == 0) {
  warning("No result files found! Check your directory path.")
} else {
  full_data <- files %>% 
    map_df(~read_csv(., show_col_types = FALSE))
  cat(paste("Loaded", length(files), "files. Total rows:", nrow(full_data), "\n"))
}

#------------------
# Helpers
#------------------
format_metric <- function(val_mean, val_se) {
  if (is.na(val_mean)) return("-")
  sprintf("%.3f (%.3f)", val_mean, val_se)
}

# Helper to parse Python list string like "[np.float64(0.9), ...]" and get Min
parse_min_lc <- function(str_val) {
  if (is.na(str_val)) return(NA)
  # Extract numeric values using regex
  nums <- str_extract_all(str_val, "\\d+\\.\\d+")[[1]]
  if (length(nums) == 0) return(NA)
  min(as.numeric(nums))
}

#------------------
# Table Generation 
#------------------
create_mnist_table <- function(df, target_alpha, save_dir = NULL, plot_conditional = FALSE) {
  
  # 1. Filter Data by Alpha (keep NA alphas for Vanilla baselines)
  dat <- df %>% 
    filter(abs(alpha - target_alpha) < 1e-6 | is.na(alpha))
  
  if (nrow(dat) == 0) return(NULL)
  
  # 2. Pre-process: Calculate Min Class Coverage from string column
  dat <- dat %>%
    rowwise() %>%
    mutate(min_class_cov = parse_min_lc(lc_covs)) %>%
    ungroup()
  
  # --- A. Set-Valued Metrics ---
  # 1. Marginal Sets (marginal = TRUE)
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
  
  # 2. Conditional Sets (marginal = FALSE) -> Use Min Class Coverage
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
  # Includes Conformal point predictions AND Vanilla baselines
  point_metrics <- dat %>%
    filter(exp == "point_valued") %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      acc_u = mean(point_acc, na.rm=TRUE), 
      acc_se = sd(point_acc, na.rm=TRUE) / sqrt(n)
    )
  
  # --- C. OOD Metrics ---
  ood_metrics <- dat %>%
    filter(exp == "ood", marginal == TRUE) %>%
    group_by(score_type) %>%
    summarise(
      n = n(),
      ood_u = mean(ood_auroc, na.rm=TRUE), 
      ood_se = sd(ood_auroc, na.rm=TRUE) / sqrt(n)
    )
  
  # --- D. Merge & Format ---
  all_scores <- unique(dat$score_type)
  base <- tibble(score_type = all_scores)
  
  joined <- base %>%
    left_join(set_marg, by="score_type") %>%
    left_join(set_cond, by="score_type") %>%
    left_join(point_metrics, by="score_type") %>%
    left_join(ood_metrics, by="score_type")
  
  formatted <- joined %>%
    mutate(
      `Marg Cov` = mapply(format_metric, m_cov_u, m_cov_se),
      `Marg Size` = mapply(format_metric, m_siz_u, m_siz_se),
      `Cond MinCov` = mapply(format_metric, c_mincov_u, c_mincov_se),
      `Cond Size` = mapply(format_metric, c_siz_u, c_siz_se),
      `Accuracy` = mapply(format_metric, acc_u, acc_se),
      `AUROC` = mapply(format_metric, ood_u, ood_se)
    )
  
  # --- E. Rename & Reorder ---
  formatted <- formatted %>%
    mutate(Method = case_when(
      score_type == "vanilla_full" ~ "Vanilla (Full)",
      score_type == "sim" ~ "Sim",
      score_type == "ratio" ~ "Ratio",
      score_type == "discount" ~ "Discount",
      score_type == "penalized" ~ "Penalized",
      score_type == "inverse_quantile" ~ "Inv-Quantile",
      TRUE ~ score_type
    )) %>%
    # preferred order
    arrange(factor(Method, levels = c("Vanilla (Full)", "Sim", "Ratio", 
                                      "Discount", "Penalized", "Inv-Quantile"))) %>%
    select(Method, everything(), -score_type)
  
  # --- F. Construct LaTeX Header ---
  additor <- list()
  additor$pos <- list(0)
  
  if (plot_conditional) {
    # Long Table with Conditional Columns
    final_df <- formatted %>% select(Method, `Marg Cov`, `Marg Size`, `Cond MinCov`, `Cond Size`, Accuracy, AUROC)
    align_str <- "ll|cccc|c|c|"
    
    additor$command <- paste0(
      "\\hline\n",
      " & \\multicolumn{4}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
      "\\cline{2-7}\n",
      "\\textbf{Method} & \\textbf{Marg Cov} & \\textbf{Marg Size} & \\textbf{Cond MinCov} & \\textbf{Cond Size} & \\textbf{Accuracy} & \\textbf{AUROC} \\\\\n",
      "\\hline\n"
    )
  } else {
    # Short Table (Marginal only)
    final_df <- formatted %>% 
      select(Method, `Marg Cov`, `Marg Size`, Accuracy, AUROC) %>%
      rename(Cov = `Marg Cov`, Size = `Marg Size`)
    align_str <- "ll|cc|c|c|"
    
    additor$command <- paste0(
      "\\hline\n",
      " & \\multicolumn{2}{c|}{\\textbf{Set-Valued}} & \\multicolumn{1}{c|}{\\textbf{Point}} & \\multicolumn{1}{c|}{\\textbf{OOD}} \\\\\n",
      "\\cline{2-5}\n",
      "\\textbf{Method} & \\textbf{Cov} & \\textbf{Size} & \\textbf{Accuracy} & \\textbf{AUROC} \\\\\n",
      "\\hline\n"
    )
  }
  
  # --- G. Print/Save ---
  ltx <- xtable(final_df, 
                caption = paste0("MNIST Results ($\\alpha=", target_alpha, "$). Values are Mean (SE)."),
                align = align_str)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    fname_suffix <- if(plot_conditional) "_cond" else ""
    filename <- file.path(save_dir, paste0("mnist_table_alpha", target_alpha, fname_suffix, ".tex"))
    
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

create_mnist_table(full_data, target_alpha = 0.05, plot_conditional = FALSE)


df_pt <- full_data %>% 
  filter(exp == "point_valued")

# Function to parse Python list strings "[np.float64(0.1), ...]" into a tidy DF
parse_python_list <- function(str_val) {
  if (is.na(str_val)) return(numeric(0))
  # Regex to extract floating point numbers
  as.numeric(unlist(str_extract_all(str_val, "\\d+\\.\\d+")))
}

# Expand the dataframe: One row per Class per Seed per Method
plot_data <- df_pt %>%
  rowwise() %>%
  mutate(acc_list = list(parse_python_list(lc_accs))) %>%
  ungroup() %>%
  select(score_type, random_state, acc_list) %>%
  unnest_longer(acc_list, indices_to = "class_idx") %>%
  rename(Accuracy = acc_list, Class = class_idx, Method = score_type) %>%
  mutate(Class = Class - 1) # Adjust to 0-indexed labels if preferred


agg_data <- plot_data %>%
  group_by(Method, Class) %>%
  summarise(Mean_Acc = mean(Accuracy), .groups = "drop")

# Define Baseline
baseline_name <- "vanilla_full"

# Calculate Difference (Method - Baseline)
baseline_vals <- agg_data %>% 
  filter(Method == baseline_name) %>% 
  select(Class, Base_Acc = Mean_Acc)

diff_data <- agg_data %>%
  left_join(baseline_vals, by = "Class") %>%
  mutate(Diff = Mean_Acc - Base_Acc) %>%
  # Clean names for plotting
  mutate(Method_Label = case_when(
    Method == "vanilla_full" ~ "Vanilla (Full)",
    Method == "vanilla_train" ~ "Vanilla (Train)",
    Method == "inverse_quantile" ~ "Inv-Quantile",
    TRUE ~ str_to_title(Method)
  ))

#-------------------------------------------------------------------------------
# 3. Visualization
#-------------------------------------------------------------------------------
# Plot A: Absolute Accuracy
p1 <- ggplot(diff_data, aes(x = factor(Class), y = Method_Label, fill = Mean_Acc)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", Mean_Acc)), size = 3) +
  scale_fill_viridis_c(name = "Acc") +
  labs(title = "Absolute Accuracy", x = "Class", y = "Method") +
  theme_minimal() + theme(panel.grid = element_blank())

# Plot B: Difference vs Vanilla (Red = Worse, Green = Better)
limit <- max(abs(diff_data$Diff), na.rm = TRUE) * 1.05

p2 <- ggplot(diff_data, aes(x = factor(Class), y = Method_Label, fill = Diff)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%+.1f%%", Diff*100)), size = 3) +
  scale_fill_gradient2(low = "#d73027", mid = "white", high = "#1a9850", 
                       midpoint = 0, limit = c(-limit, limit),
                       name = "Diff") +
  labs(title = paste("Diff vs", baseline_name), x = "Class", y = "") +
  theme_minimal() + 
  theme(panel.grid = element_blank(), axis.text.y = element_blank())

# Display side-by-side
grid.arrange(p1, p2, ncol = 2, widths = c(1, 0.9))
