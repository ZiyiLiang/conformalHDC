# Generate OOD ROC figures and comparison tables in R from diagnostic exports.

DATASETS <- c(mnist = "MNIST", fashion_mnist = "Fashion", pamap2 = "PAMAP2", har = "UCI HAR")
METHODS <- c(maxsim = "Max similarity", discount = "CHDC-discount", energy = "Energy")
NOMINAL_ALPHA <- c(mnist = 0.05, fashion_mnist = 0.05, pamap2 = 0.1, har = 0.1)

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

write_auc_table <- function(data, opts, path) {
  ids <- opts$datasets
  methods <- names(METHODS)
  means <- vapply(data[ids], function(x) {
    vapply(methods, function(method) mean(x$auroc$ood_auroc[x$auroc$method == method]), numeric(1))
  }, numeric(length(methods)))
  best <- apply(round(means, 3L), 2L, max)
  row_end <- paste0(" ", strrep("\\", 2L))
  lines <- c("% Generated from diagnostic exports by experiments_ood/make_plots.R.",
             "% Cells are mean (standard error) across repetitions; bold marks the highest displayed mean.",
             paste0("\\begin{tabular}{@{}l", strrep("c", length(ids)), "@{}}"), "\\toprule",
             partial_banner(length(ids) + 1L, paste(DATASETS[ids], collapse = ", "), opts$allow_partial),
             paste0("Method & ", paste(sprintf("\\textbf{%s}", DATASETS[ids]), collapse = " & "), row_end),
             "\\midrule",
             paste0("\\multicolumn{", length(ids) + 1L,
                    "}{c}{\\textit{OOD detection AUC $\\uparrow$}}", row_end))
  for (i in seq_along(methods)) {
    method <- methods[[i]]
    cells <- vapply(data[ids], function(x) metric_cell(x$auroc$ood_auroc[x$auroc$method == method]), character(1))
    winners <- round(means[i, ], 3L) == best
    cells[winners] <- paste0("\\textbf{", cells[winners], "}")
    lines <- c(lines, paste0(METHODS[[method]], " & ", paste(cells, collapse = " & "), row_end))
  }
  writeLines(c(lines, "\\botrule", "\\end{tabular}"), path)
}

write_operating_table <- function(dataset, data, opts, path) {
  grid <- data$grid
  n <- length(data$states)
  lines <- c("% Rejection rates are fractions; parentheses give standard errors across repetitions.",
             "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             partial_banner(7L, paste0(DATASETS[[dataset]], "; ", n, " repetitions"), opts$allow_partial),
             " & \\multicolumn{2}{c}{Max similarity} & \\multicolumn{2}{c}{CHDC-discount} & \\multicolumn{2}{c}{Energy} \\\\",
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


roc_script_path <- function() {
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg) != 1L) stop("Run with Rscript experiments_ood/make_plots.R")
  normalizePath(sub("^--file=", "", arg), mustWork = TRUE)
}

roc_path_inside <- function(path, parent) {
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  parent <- normalizePath(parent, winslash = "/", mustWork = FALSE)
  if (.Platform$OS.type == "windows") {
    path <- tolower(path)
    parent <- tolower(parent)
  }
  identical(path, parent) || startsWith(path, paste0(sub("/+$", "", parent), "/"))
}

parse_roc_options <- function(args, script) {
  repo <- dirname(dirname(script))
  opts <- list(results_root = Sys.getenv("CHDC_RESULTS_ROOT", unset = file.path(repo, "results")),
               datasets = c("mnist", "fashion_mnist", "pamap2", "har"),
               expected_repetitions = 100L, allow_partial = FALSE,
               score_diagnostics = FALSE, output_dir = NULL)
  i <- 1L
  while (i <= length(args)) {
    flag <- args[[i]]
    if (flag %in% c("--help", "-h")) {
      cat(paste0("Usage: Rscript experiments_ood/make_plots.R [options]\n",
                 "  --results-root PATH       Raw results base (default: CHDC_RESULTS_ROOT or repo/results)\n",
                 "                            Diagnostic jobs are under PATH/ood_diagnostics/ood_<dataset>\n",
                 "  --datasets IDS            Comma-separated mnist,fashion_mnist,pamap2,har\n",
                 "  --expected-repetitions N  Full-run states 1..N (default: 100)\n",
                 "  --output-dir PATH         Default: results/plots/ood_diagnostics\n",
                 "  --allow-partial           Explicitly labelled local/smoke-test report\n",
                 "  --score-diagnostics       Also plot score CDFs for the first available repetition\n"))
      return(NULL)
    }
    if (flag %in% c("--allow-partial", "--score-diagnostics")) {
      if (flag == "--allow-partial") opts$allow_partial <- TRUE
      if (flag == "--score-diagnostics") opts$score_diagnostics <- TRUE
      i <- i + 1L
      next
    }
    if (!flag %in% c("--results-root", "--datasets", "--expected-repetitions", "--output-dir")) {
      stop("Unknown argument: ", flag)
    }
    if (i == length(args)) stop("Missing value for ", flag)
    value <- args[[i + 1L]]
    if (flag == "--results-root") opts$results_root <- value
    if (flag == "--datasets") opts$datasets <- strsplit(value, ",", fixed = TRUE)[[1L]]
    if (flag == "--output-dir") opts$output_dir <- value
    if (flag == "--expected-repetitions") {
      n <- suppressWarnings(as.integer(value))
      if (is.na(n) || n < 1L || as.character(n) != value) stop("Invalid --expected-repetitions")
      opts$expected_repetitions <- n
    }
    i <- i + 2L
  }
  ids <- c("mnist", "fashion_mnist", "pamap2", "har")
  if (!length(opts$datasets) || anyDuplicated(opts$datasets) || !all(opts$datasets %in% ids)) {
    stop("Unknown or duplicate dataset ID")
  }
  if (!opts$allow_partial && !setequal(opts$datasets, ids)) {
    stop("A dataset subset requires --allow-partial")
  }
  opts$results_root <- normalizePath(opts$results_root, winslash = "/", mustWork = TRUE)
  if (is.null(opts$output_dir)) opts$output_dir <- file.path(opts$results_root, "plots", "ood_diagnostics")
  opts$output_dir <- normalizePath(opts$output_dir, winslash = "/", mustWork = FALSE)
  permitted <- file.path(opts$results_root, "plots", "ood_diagnostics")
  if ((roc_path_inside(opts$output_dir, opts$results_root) || roc_path_inside(opts$output_dir, repo)) &&
      !roc_path_inside(opts$output_dir, permitted)) {
    stop("Within the repository/results tree, output must stay under plots/ood_diagnostics")
  }
  opts
}

read_roc_export <- function(job, kind, expected_name, helper) {
  marker <- job$marker
  filename <- as.character(marker$files[[kind]])
  if (length(filename) != 1L || is.na(filename) || filename != expected_name) {
    stop(job$marker_file, ": unexpected diagnostic ", kind, " filename")
  }
  path <- file.path(dirname(job$marker_file), filename)
  expected_hash <- as.character(marker$sha256[[kind]])
  if (!file.exists(path) || length(expected_hash) != 1L || is.na(expected_hash) ||
      tolower(helper$sha256_file(path)) != tolower(expected_hash)) {
    stop(job$marker_file, ": missing or checksum-invalid diagnostic ", kind, " export")
  }
  connection <- gzfile(path, open = "rt")
  on.exit(close(connection))
  list(path = path, sha256 = expected_hash,
       data = read.csv(connection, stringsAsFactors = FALSE, check.names = FALSE))
}

validate_score_export <- function(frame, job, dataset, helper) {
  helper$require_columns(frame, c("dataset", "random_state", "partition", "sample_index", "label",
                                  names(helper$METHODS)), job$marker_file)
  states <- as.integer(job$marker$random_states)
  if (anyNA(frame[c("dataset", "random_state", "partition", "sample_index", "label")]) ||
      !all(frame$dataset == dataset) || !setequal(frame$random_state, states) ||
      !setequal(frame$partition, c("calibration", "test", "ood")) ||
      any(!is.finite(frame$sample_index)) || any(frame$sample_index < 0) ||
      any(frame$sample_index != as.integer(frame$sample_index)) ||
      any(!is.finite(frame$label)) || any(frame$label != as.integer(frame$label)) ||
      any(!is.finite(as.matrix(frame[names(helper$METHODS)])))) {
    stop(job$marker_file, ": invalid per-observation diagnostic scores")
  }
  if (anyDuplicated(paste(frame$random_state, frame$sample_index))) {
    stop(job$marker_file, ": repeated sample index within a repetition or overlapping partitions")
  }
  for (state in states) {
    counts <- table(factor(frame$partition[frame$random_state == state],
                           levels = c("calibration", "test", "ood")))
    metric <- job$auroc[job$auroc$random_state == state, ][1L, ]
    if (!identical(as.integer(counts), as.integer(unlist(metric[c("n_cal", "n_test", "n_ood")])))) {
      stop(job$marker_file, ": per-observation export and metric split counts disagree")
    }
  }
  invisible(TRUE)
}

validate_one_roc <- function(curve, expected_auc, marker_file) {
  fpr <- curve$id_false_rejection
  tpr <- curve$ood_rejection
  threshold <- curve$detection_threshold
  if (length(fpr) < 2L || any(!is.finite(fpr)) || any(!is.finite(tpr)) ||
      any(fpr < 0 | fpr > 1 | tpr < 0 | tpr > 1) ||
      any(diff(fpr) < -1e-12) || any(diff(tpr) < -1e-12) ||
      any(abs(c(fpr[1L], tpr[1L], tail(fpr, 1L) - 1, tail(tpr, 1L) - 1)) > 1e-12) ||
      !is.infinite(threshold[1L]) || threshold[1L] < 0 ||
      any(!is.finite(threshold[-1L])) || any(diff(threshold) >= 0) ||
      anyDuplicated(paste(fpr, tpr))) {
    stop(marker_file, ": invalid ROC coordinate ordering, endpoints, or thresholds")
  }
  # Check that the exported ROC integrates to its reported AUC.
  integrated <- sum(diff(fpr) * (head(tpr, -1L) + tail(tpr, -1L)) / 2)
  if (abs(integrated - expected_auc) > 1e-10) {
    stop(marker_file, ": full ROC coordinates and diagnostic AUROC disagree")
  }
  invisible(TRUE)
}

interpolate_roc <- function(fpr, tpr, grid) {
  # Match ordered-vertex interpolation (including vertical ROC segments).
  # findInterval selects the last vertex at or below each grid point. Its
  # following vertex is the first one at the next larger FPR; collapsing
  # duplicate FPRs beforehand would replace the correct approach to a jump.
  left <- findInterval(grid, fpr)
  values <- numeric(length(grid))
  below <- left == 0L
  above <- left >= length(fpr)
  values[below] <- tpr[1L]
  values[above] <- tpr[length(tpr)]
  inside <- which(!below & !above)
  if (length(inside)) {
    lo <- left[inside]
    hi <- lo + 1L
    weight <- (grid[inside] - fpr[lo]) / (fpr[hi] - fpr[lo])
    values[inside] <- tpr[lo] + weight * (tpr[hi] - tpr[lo])
  }
  values
}

load_roc_dataset <- function(dataset, opts, helper, grid) {
  metric_opts <- opts
  metric_opts$results_root <- file.path(opts$results_root, "ood_diagnostics")
  data <- helper$load_dataset(dataset, metric_opts)
  curves <- setNames(lapply(names(helper$METHODS), function(method) list()), names(helper$METHODS))
  first_scores <- NULL
  input_exports <- list()
  for (job in data$jobs) {
    if (!identical(as.integer(job$marker$diagnostic_schema_version), 1L)) {
      stop(job$marker_file, ": missing or unsupported diagnostic schema")
    }
    if (!identical(as.character(job$marker$roc_positive_class), "ood") ||
        !identical(as.character(job$marker$roc_score), "-id_score") ||
        !identical(as.character(job$marker$threshold_rule), "id_score < threshold")) {
      stop(job$marker_file, ": incompatible ROC direction or calibrated rejection convention")
    }
    group <- as.integer(job$marker$seed_group)
    roc_input <- read_roc_export(job, "roc", paste0("seed", group, "_roc.csv.gz"), helper)
    score_input <- read_roc_export(job, "scores", paste0("seed", group, "_scores.csv.gz"), helper)
    frame <- roc_input$data
    helper$require_columns(frame, c("dataset", "random_state", "method", "id_false_rejection",
                                    "ood_rejection", "detection_threshold"), roc_input$path)
    states <- as.integer(job$marker$random_states)
    if (anyNA(frame[c("dataset", "random_state", "method")]) || !all(frame$dataset == dataset) ||
        !setequal(frame$random_state, states) || !setequal(frame$method, names(helper$METHODS))) {
      stop(job$marker_file, ": ROC dataset, state, or method mismatch")
    }
    validate_score_export(score_input$data, job, dataset, helper)
    if (min(states) == min(data$states)) {
      first_scores <- score_input$data[score_input$data$random_state == min(data$states), ]
    }
    for (state in states) {
      for (method in names(helper$METHODS)) {
        curve <- frame[frame$random_state == state & frame$method == method, ]
        auc <- job$auroc$ood_auroc[job$auroc$random_state == state & job$auroc$method == method]
        validate_one_roc(curve, auc, job$marker_file)
        curves[[method]][[as.character(state)]] <- interpolate_roc(
          curve$id_false_rejection, curve$ood_rejection, grid
        )
      }
    }
    input_exports[[length(input_exports) + 1L]] <- list(
      marker_file = job$marker_file, marker_sha256 = helper$sha256_file(job$marker_file),
      files = list(roc = list(path = roc_input$path, sha256 = roc_input$sha256),
                   scores = list(path = score_input$path, sha256 = score_input$sha256),
                   auroc = list(path = job$input_files[["auroc"]], sha256 = job$marker$sha256$auroc),
                   operating = list(path = job$input_files[["operating"]], sha256 = job$marker$sha256$operating))
    )
  }
  mean_curves <- do.call(rbind, lapply(names(helper$METHODS), function(method) {
    matrix <- do.call(cbind, curves[[method]][as.character(data$states)])
    n <- ncol(matrix)
    spread <- if (n > 1L) apply(matrix, 1L, stats::sd) else rep(0, length(grid))
    mean_tpr <- rowMeans(matrix)
    data.frame(dataset = dataset, method = method, n = n, id_false_rejection = grid,
               ood_rejection_mean = mean_tpr, ood_rejection_sd = spread,
               ood_rejection_se = if (n > 1L) spread / sqrt(n) else NA_real_,
               lower_sd = pmax(0, mean_tpr - spread), upper_sd = pmin(1, mean_tpr + spread))
  }))
  nominal <- do.call(rbind, lapply(names(helper$METHODS), function(method) {
    rows <- data$operating[data$operating$method == method &
                             abs(data$operating$alpha - helper$NOMINAL_ALPHA[[dataset]]) < 1e-12, ]
    if (nrow(rows) != length(data$states)) stop(dataset, ": nominal alpha is absent from some jobs")
    auc <- helper$summarize_values(data$auroc$ood_auroc[data$auroc$method == method])
    id <- helper$summarize_values(rows$id_false_rejection)
    ood <- helper$summarize_values(rows$ood_rejection)
    data.frame(dataset = dataset, method = method, alpha = helper$NOMINAL_ALPHA[[dataset]],
               n = length(data$states), auroc_mean = auc[["mean"]], auroc_se = auc[["se"]],
               id_false_rejection_mean = id[["mean"]], id_false_rejection_se = id[["se"]],
               ood_rejection_mean = ood[["mean"]], ood_rejection_se = ood[["se"]])
  }))
  # The figure uses the smallest saved HAR alpha to avoid saturated markers.
  # Tables use the nominal alpha.
  display_points <- nominal
  display_alpha <- if (dataset == "har") min(data$grid) else helper$NOMINAL_ALPHA[[dataset]]
  for (method in names(helper$METHODS)) {
    rows <- data$operating[data$operating$method == method &
                            abs(data$operating$alpha - display_alpha) < 1e-12, ]
    if (nrow(rows) != length(data$states)) stop(dataset, ": figure alpha is absent from some jobs")
    index <- display_points$method == method
    display_points$alpha[index] <- display_alpha
    display_points$id_false_rejection_mean[index] <- mean(rows$id_false_rejection)
    display_points$ood_rejection_mean[index] <- mean(rows$ood_rejection)
  }
  list(metrics = data, mean_curves = mean_curves, nominal = nominal, display_points = display_points,
       first_scores = first_scores, input_exports = input_exports)
}

roc_palette <- c(maxsim = "#0072B2", discount = "#972D15", energy = "#009E73")

roc_inset_settings <- function(data) {
  dataset <- unique(data$mean_curves$dataset)
  if (length(dataset) != 1L) stop("Inset requires exactly one dataset")
  xlim <- switch(dataset, mnist = c(0, 0.10), fashion_mnist = c(0, 0.06),
                 pamap2 = c(0, 0.15), har = {
                   # Include the displayed calibrated points with 10% padding,
                   # rounded up to the next hundredth.
                   c(0, ceiling(1.1 * max(data$display_points$id_false_rejection_mean) * 100) / 100)
                 })
  if (is.null(xlim)) stop("Unknown dataset for ROC inset: ", dataset)
  visible <- data$mean_curves[data$mean_curves$id_false_rejection >= xlim[1L] &
                                data$mean_curves$id_false_rejection <= xlim[2L], ]
  ylim <- if (dataset == "fashion_mnist") {
    # Magnify the operating points rather than the high-rejection plateau.
    point_range <- range(data$display_points$ood_rejection_mean)
    padding <- max(0.005, 0.25 * diff(point_range))
    c(0,
      min(1, ceiling((point_range[2L] + padding) * 100) / 100))
  } else if (dataset %in% c("pamap2", "har")) c(0, 1) else
    c(max(0, floor(min(visible$lower_sd) * 20) / 20),
      min(1, ceiling(max(visible$upper_sd) * 20) / 20))
  xticks <- switch(dataset, mnist = c(0, 0.05, 0.10), fashion_mnist = c(0, 0.02, 0.04, 0.06),
                   pamap2 = c(0, 0.05, 0.10, 0.15), har = pretty(xlim, n = 3L))
  xticks <- xticks[xticks >= xlim[1L] & xticks <= xlim[2L]]
  yticks <- pretty(ylim, n = 2L)
  yticks <- yticks[yticks >= ylim[1L] & yticks <= ylim[2L]]
  list(xlim = xlim, ylim = ylim, xticks = xticks, yticks = yticks)
}

draw_roc_curves <- function(data, inset = FALSE) {
  if (inset) {
    settings <- roc_inset_settings(data)
    bounds <- c(left = 0.53, right = 0.97, bottom = 0.12, top = 0.53)
    graphics::rect(0.455, 0.055, 0.995, 0.595, col = "white", border = "grey80")
    map_x <- function(x) bounds[["left"]] + (x - settings$xlim[1L]) / diff(settings$xlim) *
      (bounds[["right"]] - bounds[["left"]])
    map_y <- function(y) bounds[["bottom"]] + (y - settings$ylim[1L]) / diff(settings$ylim) *
      (bounds[["top"]] - bounds[["bottom"]])
    original_usr <- graphics::par("usr")
    graphics::clip(bounds[["left"]], bounds[["right"]], bounds[["bottom"]], bounds[["top"]])
    for (method in names(roc_palette)) {
      curve <- data$mean_curves[data$mean_curves$method == method &
                                 data$mean_curves$id_false_rejection >= settings$xlim[1L] &
                                 data$mean_curves$id_false_rejection <= settings$xlim[2L], ]
      graphics::polygon(c(map_x(curve$id_false_rejection), rev(map_x(curve$id_false_rejection))),
                        c(map_y(curve$lower_sd), rev(map_y(curve$upper_sd))),
                        border = NA, col = grDevices::adjustcolor(roc_palette[[method]], alpha.f = 0.14))
      graphics::lines(map_x(curve$id_false_rejection), map_y(curve$ood_rejection_mean),
                      col = roc_palette[[method]], lwd = 1.4)
    }
    # Draw every marker last so subsequent curves or bands cannot cover it.
    for (method in names(roc_palette)) {
      point <- data$display_points[data$display_points$method == method, ]
      if (point$id_false_rejection_mean >= settings$xlim[1L] &&
          point$id_false_rejection_mean <= settings$xlim[2L] &&
          point$ood_rejection_mean >= settings$ylim[1L] &&
          point$ood_rejection_mean <= settings$ylim[2L]) {
        graphics::points(map_x(point$id_false_rejection_mean), map_y(point$ood_rejection_mean),
                         pch = 19, col = roc_palette[[method]], cex = 0.65)
      }
    }
    graphics::clip(original_usr[1L], original_usr[2L], original_usr[3L], original_usr[4L])
    x_ticks <- settings$xticks
    y_ticks <- settings$yticks
    graphics::rect(bounds[["left"]], bounds[["bottom"]], bounds[["right"]], bounds[["top"]], border = "grey35")
    graphics::segments(map_x(x_ticks), bounds[["bottom"]], map_x(x_ticks), bounds[["bottom"]] - 0.008)
    graphics::segments(bounds[["left"]], map_y(y_ticks), bounds[["left"]] - 0.008, map_y(y_ticks))
    graphics::text(map_x(x_ticks), bounds[["bottom"]] - 0.027,
                   labels = format(x_ticks, trim = TRUE, scientific = FALSE), cex = 0.63)
    graphics::text(bounds[["left"]] - 0.018, map_y(y_ticks),
                   labels = format(y_ticks, trim = TRUE, scientific = FALSE), adj = 1, cex = 0.63)
    return(invisible(NULL))
  }
  for (method in names(roc_palette)) {
    curve <- data$mean_curves[data$mean_curves$method == method, ]
    graphics::polygon(c(curve$id_false_rejection, rev(curve$id_false_rejection)),
                      c(curve$lower_sd, rev(curve$upper_sd)), border = NA,
                      col = grDevices::adjustcolor(roc_palette[[method]], alpha.f = 0.14))
    graphics::lines(curve$id_false_rejection, curve$ood_rejection_mean, col = roc_palette[[method]], lwd = 1.8)
    point <- data$display_points[data$display_points$method == method, ]
    graphics::points(point$id_false_rejection_mean, point$ood_rejection_mean,
                     pch = 19, cex = 0.85, col = roc_palette[[method]])
  }
}

write_roc_pdf <- function(data, opts, helper, path) {
  n <- length(data)
  layout <- if (n == 1L) c(1, 1) else if (n == 2L) c(1, 2) else c(2, 2)
  grDevices::pdf(path, width = if (n == 1L) 6 else 11.2,
                 height = if (n <= 2L) 5.5 else 9, family = "Helvetica", useDingbats = FALSE)
  on.exit(grDevices::dev.off())
  graphics::par(mfrow = layout, mar = c(4, 4.2, 2.4, 0.7), oma = c(2, 0, 0.3, 0),
                mgp = c(2.5, 0.7, 0), las = 1, cex = 0.88)
  for (dataset in names(data)) {
    value <- data[[dataset]]
    title <- paste0(if (opts$allow_partial) "PARTIAL: " else "", helper$DATASETS[[dataset]])
    graphics::plot(NA_real_, NA_real_, xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i",
                   xlab = "ID false rejection rate", ylab = "OOD rejection rate", main = title)
    graphics::abline(0, 1, col = "grey75", lty = 3)
    draw_roc_curves(value)
    draw_roc_curves(value, inset = TRUE)
  }
  graphics::mtext(unname(helper$METHODS[names(roc_palette)]), side = 1, outer = TRUE,
                  at = c(0.25, 0.5, 0.75), col = unname(roc_palette), line = 0.25, font = 2, cex = 0.95)
}

write_score_cdf_pdf <- function(data, opts, helper, path) {
  grDevices::pdf(path, width = 8.2, height = 5.8, family = "Helvetica", useDingbats = FALSE)
  on.exit(grDevices::dev.off())
  graphics::par(mar = c(4.4, 4.3, 4, 1), mgp = c(2.5, 0.7, 0), las = 1)
  for (dataset in names(data)) {
    value <- data[[dataset]]
    scores <- value$first_scores
    state <- min(value$metrics$states)
    labels <- sort(unique(scores$label[scores$partition == "ood"]))
    colors <- c("black", "grey45", grDevices::hcl.colors(length(labels), palette = "Dark 3"))
    for (method in names(helper$METHODS)) {
      groups <- c(list(scores[[method]][scores$partition == "calibration"],
                       scores[[method]][scores$partition == "test"]),
                  lapply(labels, function(label) scores[[method]][scores$partition == "ood" & scores$label == label]))
      limits <- range(unlist(groups, use.names = FALSE))
      if (diff(limits) == 0) limits <- limits + c(-0.01, 0.01)
      graphics::plot(NA_real_, NA_real_, xlim = limits, ylim = c(0, 1),
                     xlab = paste0(helper$METHODS[[method]], " score (larger = ID)"),
                     ylab = "Empirical cumulative probability",
                     main = paste0(if (opts$allow_partial) "PARTIAL: " else "", helper$DATASETS[[dataset]],
                                   " - ", helper$METHODS[[method]], "; repetition ", state))
      for (i in seq_along(groups)) {
        graphics::lines(stats::ecdf(groups[[i]]), do.points = FALSE, verticals = TRUE,
                        col = colors[[i]], lty = if (i == 2L) 2 else 1, lwd = if (i <= 2L) 2 else 1.4)
      }
      row <- value$metrics$operating[value$metrics$operating$random_state == state &
                                       value$metrics$operating$method == method &
                                       abs(value$metrics$operating$alpha - helper$NOMINAL_ALPHA[[dataset]]) < 1e-12, ]
      if (is.finite(row$threshold)) graphics::abline(v = row$threshold, col = roc_palette[[method]], lty = 3, lwd = 1.8)
      graphics::legend("topleft", legend = c("ID calibration", "ID test", paste("OOD class", labels)),
                       col = colors, lty = c(1, 2, rep(1, length(labels))), lwd = c(2, 2, rep(1.4, length(labels))),
                       bty = "n", cex = 0.85)
      graphics::mtext(paste0("First available repetition; dotted line: ID-calibrated cutoff, alpha = ", row$alpha,
                             ". Rejection uses score < cutoff."), side = 3, line = 0.5, cex = 0.70)
    }
  }
}

write_nominal_roc_table <- function(summary, opts, helper, path) {
  cell <- function(mean, se) if (is.na(se)) sprintf("%.3f (n/a)", mean) else sprintf("%.3f (%.3f)", mean, se)
  lines <- c("% Generated by experiments_ood/make_plots.R from diagnostic exports.",
             "% Mean (standard error) across repetitions; rejection rates are fractions.",
             "\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
             helper$partial_banner(6L, paste(helper$DATASETS[opts$datasets], collapse = ", "), opts$allow_partial),
             "Dataset & Method & $\\alpha$ & AUC & ID FR & OOD rejection \\\\", "\\midrule")
  for (i in seq_len(nrow(summary))) {
    row <- summary[i, ]
    lines <- c(lines, paste0(helper$DATASETS[[row$dataset]], " & ", helper$METHODS[[row$method]], " & ",
                             format(row$alpha, trim = TRUE), " & ", cell(row$auroc_mean, row$auroc_se), " & ",
                             cell(row$id_false_rejection_mean, row$id_false_rejection_se), " & ",
                             cell(row$ood_rejection_mean, row$ood_rejection_se), " \\\\"))
  }
  writeLines(c(lines, "\\botrule", "\\end{tabular}"), path)
}

run_roc_reporting <- function(opts, script) {
  for (package in c("jsonlite", "digest")) {
    if (!requireNamespace(package, quietly = TRUE)) stop("Required R package is unavailable: ", package)
  }
  helper <- environment(run_roc_reporting)
  # Dense low-FPR sampling; each repetition is interpolated separately, never pooled.
  grid <- sort(unique(c(seq(0, 0.15, by = 0.0005), seq(0.15, 1, length.out = 1701L))))
  data <- setNames(lapply(opts$datasets, load_roc_dataset, opts = opts, helper = helper, grid = grid), opts$datasets)
  temperatures <- unique(vapply(data, function(x) unique(x$metrics$auroc$temperature), numeric(1)))
  if (!opts$allow_partial && length(temperatures) != 1L) stop("Datasets use inconsistent energy temperatures")
  # Nothing is generated until every requested diagnostic dataset has validated.
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  prefix <- if (opts$allow_partial) "partial_" else ""
  path <- function(name) file.path(opts$output_dir, paste0(prefix, name))
  mean_curves <- do.call(rbind, lapply(data, `[[`, "mean_curves"))
  nominal <- do.call(rbind, lapply(data, `[[`, "nominal"))
  write.csv(mean_curves, path("ood_mean_roc.csv"), row.names = FALSE)
  write.csv(nominal, path("ood_nominal_summary.csv"), row.names = FALSE)
  auc_summary <- nominal[c("dataset", "method", "n", "auroc_mean", "auroc_se")]
  names(auc_summary) <- c("dataset", "method", "n", "mean", "se")
  write.csv(auc_summary, path("ood_auc_summary.csv"), row.names = FALSE)
  write_nominal_roc_table(nominal, opts, helper, path("ood_nominal_summary.tex"))
  metric_data <- lapply(data, `[[`, "metrics")
  helper$write_auc_table(metric_data, opts, path("ood_summary_table.tex"))
  for (dataset in opts$datasets) {
    helper$write_operating_table(dataset, metric_data[[dataset]], opts,
                                 path(paste0("ood_operating_", dataset, ".tex")))
  }
  write_roc_pdf(data, opts, helper, path("ood_roc.pdf"))
  if (opts$score_diagnostics) write_score_cdf_pdf(data, opts, helper, path("ood_score_cdfs.pdf"))
  metadata <- list(
    generated_utc = format(Sys.time(), tz = "UTC", format = "%Y-%m-%dT%H:%M:%SZ"),
    mode = if (opts$allow_partial) "partial" else "full", datasets = opts$datasets,
    results_root = opts$results_root, diagnostic_root = file.path(opts$results_root, "ood_diagnostics"),
    observed_repetitions = lapply(data, function(x) length(x$metrics$states)),
    expected_repetitions = opts$expected_repetitions,
    report_script_sha256 = helper$sha256_file(script),
    roc_positive_class = "OOD", roc_detection_score = "negative ID score",
    averaging = "Per-repetition ROC with ordered vertical vertices retained; exact-FPR ties use the last vertex, interpolation proceeds to the first next larger-FPR vertex; arithmetic mean on a shared FPR grid",
    fpr_grid = grid, shading = "Mean +/- one standard deviation across repetitions, clipped to [0,1]; not a population confidence interval",
    auroc_summary = "Mean and standard error of individual repetition AUCs; not the area of the interpolated mean curve",
    operating_points = "Separately averaged empirical rates at ID-calibration-selected cutoffs; HAR uses its smallest saved alpha, other datasets use nominal alpha; coordinates are not forced onto the mean ROC",
    figure_operating_alpha = lapply(data, function(x) unique(x$display_points$alpha)),
    figure_insets = lapply(data, roc_inset_settings),
    score_cdf_scope = if (opts$score_diagnostics) lapply(data, function(x) min(x$metrics$states)) else NULL,
    input_exports = lapply(data, `[[`, "input_exports")
  )
  jsonlite::write_json(metadata, path("ood_roc_metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat("Saved ", if (opts$allow_partial) "explicitly PARTIAL" else "full four-benchmark",
      " OOD diagnostic report under ", opts$output_dir, "\n", sep = "")
  invisible(list(mean_curves = mean_curves, nominal = nominal, metadata = metadata))
}

if (sys.nframe() == 0L) {
  script <- roc_script_path()
  opts <- parse_roc_options(commandArgs(trailingOnly = TRUE), script)
  if (!is.null(opts)) run_roc_reporting(opts, script)
}
