# Generate OOD comparison tables directly from complete repository experiment jobs.
# No manuscript table is read as an input.

DATASETS <- c(mnist = "MNIST", fashion_mnist = "Fashion", pamap2 = "PAMAP2", har = "UCI HAR")
METHODS <- c(maxsim = "Max similarity", discount = "Discount", energy = "Energy")
NOMINAL_ALPHA <- c(mnist = 0.05, fashion_mnist = 0.05, pamap2 = 0.1, har = 0.1)

script_path <- function() {
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg) != 1L) stop("Run this generator with Rscript experiments_ood/make_plots.R")
  normalizePath(sub("^--file=", "", arg), mustWork = TRUE)
}

parse_options <- function(args, script) {
  repo <- dirname(dirname(script))
  root <- Sys.getenv("CHDC_RESULTS_ROOT", unset = file.path(repo, "results"))
  opts <- list(results_root = root, datasets = names(DATASETS), allow_partial = FALSE,
               expected_repetitions = 100L)
  i <- 1L
  while (i <= length(args)) {
    flag <- args[[i]]
    if (flag %in% c("--help", "-h")) {
      cat(paste0("Usage: Rscript experiments_ood/make_plots.R [options]\n",
                 "  --results-root PATH       Raw results root (default: CHDC_RESULTS_ROOT or repo/results)\n",
                 "  --datasets IDS            Comma-separated mnist,fashion_mnist,pamap2,har\n",
                 "  --expected-repetitions N  Required repetitions per dataset (default: 100)\n",
                 "  --allow-partial           Explicit smoke-test report; output names and tables say partial\n"))
      return(NULL)
    }
    if (flag == "--allow-partial") {
      opts$allow_partial <- TRUE
      i <- i + 1L
      next
    }
    if (!flag %in% c("--results-root", "--datasets", "--expected-repetitions")) {
      stop("Unknown argument: ", flag)
    }
    if (i == length(args)) stop("Missing value for ", flag)
    value <- args[[i + 1L]]
    if (flag == "--results-root") opts$results_root <- value
    if (flag == "--datasets") opts$datasets <- strsplit(value, ",", fixed = TRUE)[[1L]]
    if (flag == "--expected-repetitions") {
      parsed <- suppressWarnings(as.integer(value))
      if (is.na(parsed) || parsed < 1L || as.character(parsed) != value) {
        stop("--expected-repetitions must be a positive integer")
      }
      opts$expected_repetitions <- parsed
    }
    i <- i + 2L
  }
  if (!length(opts$datasets) || anyDuplicated(opts$datasets) ||
      !all(opts$datasets %in% names(DATASETS))) stop("Unknown or duplicate dataset ID")
  if (!opts$allow_partial && !setequal(opts$datasets, names(DATASETS))) {
    stop("A dataset subset requires --allow-partial; full-paper output requires all four benchmarks")
  }
  opts$results_root <- normalizePath(opts$results_root, mustWork = TRUE)
  opts$output_dir <- file.path(opts$results_root, "plots", "ood")
  opts
}

require_columns <- function(df, columns, file) {
  missing <- setdiff(columns, names(df))
  if (length(missing)) stop(file, ": missing columns: ", paste(missing, collapse = ", "))
}

sha256_file <- function(path) digest::digest(file = path, algo = "sha256")
alpha_key <- function(alpha) sprintf("%.12f", alpha)

validate_job <- function(marker_file, dataset) {
  marker <- jsonlite::fromJSON(marker_file, simplifyVector = TRUE)
  required_marker <- c("schema_version", "dataset", "seed_group", "repetitions", "random_states",
                       "methods", "alpha_grid", "nominal_alpha", "temperature", "dimension",
                       "source_hashes", "input_array_sha256", "files", "sha256")
  if (!all(required_marker %in% names(marker))) stop(marker_file, ": incomplete completion metadata")
  package_versions <- marker$run_config$package_versions
  if (is.null(package_versions) || !length(package_versions) ||
      is.null(names(package_versions)) || anyNA(names(package_versions)) ||
      any(!nzchar(names(package_versions))) || any(lengths(package_versions) != 1L) ||
      anyNA(unlist(package_versions, use.names = FALSE)) ||
      any(!nzchar(trimws(as.character(unlist(package_versions, use.names = FALSE)))))) {
    stop(marker_file, ": missing or empty numerical-library package_versions metadata")
  }
  expected_grid <- sort(as.numeric(marker$alpha_grid))
  if (!length(expected_grid) || any(!is.finite(expected_grid)) ||
      any(expected_grid <= 0 | expected_grid >= 1) || anyDuplicated(alpha_key(expected_grid))) {
    stop(marker_file, ": invalid alpha grid")
  }
  if (!identical(as.character(marker$dataset), dataset)) stop(marker_file, ": dataset mismatch")
  states <- as.integer(marker$random_states)
  if (!length(states) || anyNA(states) || any(states < 1L) || anyDuplicated(states) ||
      length(states) != as.integer(marker$repetitions)) stop(marker_file, ": invalid repetition metadata")
  if (!setequal(as.character(marker$methods), names(METHODS))) stop(marker_file, ": method mismatch")
  if (!isTRUE(all.equal(as.numeric(marker$nominal_alpha), unname(NOMINAL_ALPHA[[dataset]])))) {
    stop(marker_file, ": nominal alpha mismatch")
  }
  group <- as.integer(marker$seed_group)
  if (length(group) != 1L || is.na(group) || group < 1L) stop(marker_file, ": invalid seed group")
  expected_names <- c(auroc = paste0("seed", group, "_ood.csv"),
                      operating = paste0("seed", group, "_operating.csv"))
  inputs <- lapply(names(expected_names), function(kind) {
    name <- as.character(marker$files[[kind]])
    if (length(name) != 1L || name != expected_names[[kind]]) stop(marker_file, ": unexpected ", kind, " filename")
    path <- file.path(dirname(marker_file), name)
    if (!file.exists(path)) stop(marker_file, ": missing ", name)
    checksum <- as.character(marker$sha256[[kind]])
    if (length(checksum) != 1L || !identical(tolower(checksum), tolower(sha256_file(path)))) {
      stop(marker_file, ": checksum mismatch for ", name)
    }
    list(path = path, data = read.csv(path, stringsAsFactors = FALSE, check.names = FALSE))
  })
  names(inputs) <- names(expected_names)
  auc <- inputs$auroc$data
  op <- inputs$operating$data
  shared <- c("dataset", "random_state", "method", "nominal_alpha", "temperature", "n_cal", "n_test", "n_ood")
  require_columns(auc, c(shared, "ood_auroc", "dimension", "n_train"), inputs$auroc$path)
  require_columns(op, c(shared, "alpha", "threshold", "id_false_rejection", "ood_rejection",
                        "id_rejected", "ood_rejected"), inputs$operating$path)
  for (entry in list(auc, op)) {
    if (anyNA(entry$dataset) || !all(entry$dataset == dataset) ||
        anyNA(entry$method) || !all(entry$method %in% names(METHODS)) ||
        anyNA(entry$random_state) || !setequal(as.integer(entry$random_state), states)) {
      stop(marker_file, ": CSV dataset, method, or repetition mismatch")
    }
    if (any(!is.finite(entry$nominal_alpha)) ||
        any(abs(entry$nominal_alpha - NOMINAL_ALPHA[[dataset]]) > 1e-12) ||
        any(!is.finite(entry$temperature)) || any(entry$temperature <= 0) ||
        any(abs(entry$temperature - as.numeric(marker$temperature)) > 1e-12)) {
      stop(marker_file, ": CSV alpha/temperature mismatch")
    }
    if (any(!is.finite(as.matrix(entry[c("n_cal", "n_test", "n_ood")])) |
            as.matrix(entry[c("n_cal", "n_test", "n_ood")]) <= 0)) stop(marker_file, ": invalid split counts")
  }
  expected_auc <- expand.grid(random_state = states, method = names(METHODS), stringsAsFactors = FALSE)
  auc_keys <- paste(auc$random_state, auc$method)
  if (anyDuplicated(auc_keys) || !setequal(auc_keys, paste(expected_auc$random_state, expected_auc$method))) {
    stop(marker_file, ": missing or duplicate AUROC rows")
  }
  expected_op <- expand.grid(random_state = states, method = names(METHODS), alpha = expected_grid,
                             stringsAsFactors = FALSE)
  op_keys <- paste(op$random_state, op$method, alpha_key(op$alpha))
  if (anyDuplicated(op_keys) ||
      !setequal(op_keys, paste(expected_op$random_state, expected_op$method, alpha_key(expected_op$alpha)))) {
    stop(marker_file, ": missing or duplicate operating-point rows")
  }
  for (values in list(auc$ood_auroc, op$id_false_rejection, op$ood_rejection)) {
    if (any(!is.finite(values)) || any(values < 0 | values > 1)) stop(marker_file, ": invalid metric values")
  }
  if (anyNA(op$threshold) || any(op$id_rejected < 0 | op$id_rejected > op$n_test) ||
      any(op$ood_rejected < 0 | op$ood_rejected > op$n_ood) ||
      any(abs(op$id_false_rejection - op$id_rejected / op$n_test) > 1e-12) ||
      any(abs(op$ood_rejection - op$ood_rejected / op$n_ood) > 1e-12)) {
    stop(marker_file, ": rejection counts/rates disagree")
  }
  if (any(!is.finite(auc$dimension)) || any(auc$dimension != as.numeric(marker$dimension)) ||
      any(!is.finite(auc$n_train)) || any(auc$n_train <= 0)) {
    stop(marker_file, ": invalid training size or dimension metadata")
  }
  corresponding_auc <- match(paste(op$random_state, op$method), auc_keys)
  for (column in c("n_cal", "n_test", "n_ood")) {
    if (any(op[[column]] != auc[[column]][corresponding_auc])) {
      stop(marker_file, ": AUROC/operating split counts disagree")
    }
  }
  list(auroc = auc, operating = op, marker = marker, marker_file = marker_file,
       input_files = vapply(inputs, `[[`, character(1), "path"))
}

load_dataset <- function(dataset, opts) {
  folder <- file.path(opts$results_root, paste0("ood_", dataset))
  markers <- list.files(folder, pattern = "^seed[0-9]+_complete\\.json$", full.names = TRUE)
  if (!length(markers)) stop("No complete OOD jobs for ", dataset, " under ", folder)
  jobs <- lapply(markers, validate_job, dataset = dataset)
  signature <- function(job) {
    values <- job$marker[c("schema_version", "dataset", "repetitions", "methods",
                           "alpha_grid", "nominal_alpha", "temperature", "dimension",
                           "device", "threads", "source_hashes", "input_array_sha256")]
    versions <- job$marker$run_config$package_versions
    values$package_versions <- versions[sort(names(versions))]
    values
  }
  first_signature <- signature(jobs[[1L]])
  if (!all(vapply(jobs, function(job) identical(signature(job), first_signature), logical(1)))) {
    stop(dataset, ": completed jobs have inconsistent source, data, or experiment configurations")
  }
  auc <- do.call(rbind, lapply(jobs, `[[`, "auroc"))
  op <- do.call(rbind, lapply(jobs, `[[`, "operating"))
  if (anyDuplicated(paste(auc$random_state, auc$method)) ||
      anyDuplicated(paste(op$random_state, op$method, alpha_key(op$alpha)))) {
    stop(dataset, ": duplicate repetition IDs across completed seed groups")
  }
  states <- sort(unique(as.integer(auc$random_state)))
  if (!opts$allow_partial && !identical(states, seq_len(opts$expected_repetitions))) {
    stop(dataset, ": expected repetition states 1..", opts$expected_repetitions,
         "; found ", length(states), ". Use --allow-partial only for an explicitly labelled diagnostic report.")
  }
  temperatures <- unique(auc$temperature)
  dimensions <- unique(auc$dimension)
  if (length(temperatures) != 1L || length(dimensions) != 1L) {
    stop(dataset, ": jobs use inconsistent temperatures or dimensions")
  }
  cat(DATASETS[[dataset]], ": ", length(states), " repetitions, ", length(jobs), " complete seed groups\n", sep = "")
  list(auroc = auc, operating = op, jobs = jobs, states = states,
       grid = sort(as.numeric(jobs[[1L]]$marker$alpha_grid)))
}

summarize_values <- function(values) {
  n <- length(values)
  c(n = n, mean = mean(values), se = if (n > 1L) sd(values) / sqrt(n) else NA_real_)
}

metric_cell <- function(values) {
  stat <- summarize_values(values)
  if (is.na(stat[["se"]])) sprintf("%.3f (n/a)", stat[["mean"]])
  else sprintf("%.3f (%.3f)", stat[["mean"]], stat[["se"]])
}

partial_banner <- function(ncol, label, partial) {
  if (!partial) return(character())
  c(paste0("\\multicolumn{", ncol, "}{c}{\\textbf{Partial run: ", label, "}} \\\\"), "\\midrule")
}

write_auroc_table <- function(data, opts, path) {
  ids <- opts$datasets
  counts <- vapply(data, function(x) length(x$states), integer(1))
  labels <- paste0(DATASETS[ids], " ($n=", counts, "$")
  labels <- paste0(labels, ")")
  lines <- c("% Generated from raw OOD experiment jobs by experiments_ood/make_plots.R.",
             paste0("\\begin{tabular}{@{}l", strrep("c", length(ids)), "@{}}"), "\\toprule",
             partial_banner(length(ids) + 1L, paste(DATASETS[ids], collapse = ", "), opts$allow_partial),
             paste0("Method & ", paste(labels, collapse = " & "), " \\\\") , "\\midrule")
  for (method in names(METHODS)) {
    cells <- vapply(data, function(x) metric_cell(x$auroc$ood_auroc[x$auroc$method == method]), character(1))
    lines <- c(lines, paste0(METHODS[[method]], " & ", paste(cells, collapse = " & "), " \\\\"))
  }
  writeLines(c(lines, "\\botrule", "\\end{tabular}"), path)
}

write_operating_table <- function(dataset, data, opts, path) {
  grid <- data$grid
  n <- length(data$states)
  lines <- c("% Rejection rates are fractions; parentheses give standard errors across repetitions.",
             "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             partial_banner(7L, paste0(DATASETS[[dataset]], "; ", n, " repetitions"), opts$allow_partial),
             " & \\multicolumn{2}{c}{Max similarity} & \\multicolumn{2}{c}{Discount} & \\multicolumn{2}{c}{Energy} \\\\",
             "$\\alpha$ & ID FR $\\downarrow$ & OOD rejection $\\uparrow$ & ID FR $\\downarrow$ & OOD rejection $\\uparrow$ & ID FR $\\downarrow$ & OOD rejection $\\uparrow$ \\\\",
             "\\midrule")
  for (alpha in grid) {
    cells <- unlist(lapply(names(METHODS), function(method) {
      rows <- data$operating[data$operating$method == method &
                              abs(data$operating$alpha - alpha) < 1e-12, ]
      c(metric_cell(rows$id_false_rejection), metric_cell(rows$ood_rejection))
    }), use.names = FALSE)
    alpha_label <- format(alpha, scientific = FALSE, trim = TRUE, digits = 6)
    lines <- c(lines, paste0(alpha_label, " & ", paste(cells, collapse = " & "), " \\\\"))
  }
  writeLines(c(lines, "\\botrule", "\\end{tabular}"), path)
}

run_reporting <- function(opts, script) {
  for (package in c("jsonlite", "digest")) {
    if (!requireNamespace(package, quietly = TRUE)) {
      stop("Required R package is unavailable: ", package, ". Install it in the configured R environment before running.")
    }
  }
  # Validate every selected dataset before creating any report output.
  data <- setNames(lapply(opts$datasets, load_dataset, opts = opts), opts$datasets)
  if (!opts$allow_partial && length(unique(vapply(data, function(x) unique(x$auroc$temperature), numeric(1)))) != 1L) {
    stop("Selected datasets use inconsistent energy temperatures")
  }
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  prefix <- if (opts$allow_partial) "partial_" else ""
  write_auroc_table(data, opts, file.path(opts$output_dir, paste0(prefix, "ood_summary_table.tex")))
  for (dataset in opts$datasets) {
    write_operating_table(dataset, data[[dataset]], opts,
                          file.path(opts$output_dir, paste0(prefix, "ood_operating_", dataset, ".tex")))
  }
  auc_summary <- do.call(rbind, lapply(opts$datasets, function(dataset) {
    do.call(rbind, lapply(names(METHODS), function(method) {
      values <- data[[dataset]]$auroc$ood_auroc[data[[dataset]]$auroc$method == method]
      stat <- summarize_values(values)
      data.frame(dataset = dataset, method = method, n = stat[["n"]], mean = stat[["mean"]], se = stat[["se"]])
    }))
  }))
  op_summary <- do.call(rbind, lapply(opts$datasets, function(dataset) {
    do.call(rbind, lapply(names(METHODS), function(method) {
      do.call(rbind, lapply(data[[dataset]]$grid, function(alpha) {
        rows <- data[[dataset]]$operating[data[[dataset]]$operating$method == method &
                                          abs(data[[dataset]]$operating$alpha - alpha) < 1e-12, ]
        id <- summarize_values(rows$id_false_rejection)
        ood <- summarize_values(rows$ood_rejection)
        data.frame(dataset = dataset, method = method, alpha = alpha, n = id[["n"]],
                   id_false_rejection_mean = id[["mean"]], id_false_rejection_se = id[["se"]],
                   ood_rejection_mean = ood[["mean"]], ood_rejection_se = ood[["se"]])
      }))
    }))
  }))
  write.csv(auc_summary, file.path(opts$output_dir, paste0(prefix, "ood_auc_summary.csv")), row.names = FALSE)
  write.csv(op_summary, file.path(opts$output_dir, paste0(prefix, "ood_operating_summary.csv")), row.names = FALSE)
  provenance <- list(
    generated_utc = format(Sys.time(), tz = "UTC", format = "%Y-%m-%dT%H:%M:%SZ"),
    mode = if (opts$allow_partial) "partial" else "full", datasets = opts$datasets,
    results_root = opts$results_root, expected_repetitions = opts$expected_repetitions,
    report_script_sha256 = sha256_file(script),
    observed_repetitions = lapply(data, function(x) length(x$states)),
    input_jobs = unlist(lapply(data, function(x) lapply(x$jobs, function(job) {
      list(marker_file = job$marker_file, marker_sha256 = sha256_file(job$marker_file), marker = job$marker)
    })), recursive = FALSE)
  )
  jsonlite::write_json(provenance, file.path(opts$output_dir, paste0(prefix, "ood_report_metadata.json")),
                       auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat("Saved ", if (opts$allow_partial) "explicitly PARTIAL" else "full four-benchmark",
      " OOD report under ", opts$output_dir, "\n", sep = "")
  invisible(list(auroc = auc_summary, operating = op_summary, provenance = provenance))
}

if (sys.nframe() == 0L) {
  script <- script_path()
  opts <- parse_options(commandArgs(trailingOnly = TRUE), script)
  if (!is.null(opts)) run_reporting(opts, script)
}
