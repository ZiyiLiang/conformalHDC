# ROC figures from repository diagnostic exports. Original experiment files are read only.
# Run with Rscript; graphics are written to explicit file devices without previews.

roc_script_path <- function() {
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg) != 1L) stop("Run with Rscript experiments_ood/make_roc_plots.R")
  normalizePath(sub("^--file=", "", arg), mustWork = TRUE)
}

roc_report_helpers <- function(script) {
  helper <- file.path(dirname(script), "make_plots.R")
  if (!file.exists(helper)) stop("Missing existing repository reporting functions: ", helper)
  env <- new.env(parent = globalenv())
  sys.source(helper, envir = env)
  env
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
      cat(paste0("Usage: Rscript experiments_ood/make_roc_plots.R [options]\n",
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
  # Integrity check against the metric from this same diagnostic export only.
  # Legacy-versus-diagnostic comparisons belong in the separate temporary audit.
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
        # Retain ordered vertical vertices and interpolate each run separately.
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
  list(metrics = data, mean_curves = mean_curves, nominal = nominal,
       first_scores = first_scores, input_exports = input_exports)
}

roc_palette <- c(maxsim = "#0072B2", discount = "#D55E00", energy = "#009E73")

draw_roc_curves <- function(data, inset = FALSE) {
  if (inset) {
    bounds <- c(left = 0.53, right = 0.97, bottom = 0.12, top = 0.53)
    graphics::rect(0.455, 0.055, 0.995, 0.595, col = "white", border = "grey80")
    map_x <- function(x) bounds[["left"]] + x / 0.15 * (bounds[["right"]] - bounds[["left"]])
    map_y <- function(y) bounds[["bottom"]] + y * (bounds[["top"]] - bounds[["bottom"]])
    for (method in names(roc_palette)) {
      curve <- data$mean_curves[data$mean_curves$method == method & data$mean_curves$id_false_rejection <= 0.15, ]
      graphics::polygon(c(map_x(curve$id_false_rejection), rev(map_x(curve$id_false_rejection))),
                        c(map_y(curve$lower_sd), rev(map_y(curve$upper_sd))),
                        border = NA, col = grDevices::adjustcolor(roc_palette[[method]], alpha.f = 0.14))
      graphics::lines(map_x(curve$id_false_rejection), map_y(curve$ood_rejection_mean),
                      col = roc_palette[[method]], lwd = 1.4)
      point <- data$nominal[data$nominal$method == method, ]
      if (point$id_false_rejection_mean <= 0.15) {
        graphics::points(map_x(point$id_false_rejection_mean), map_y(point$ood_rejection_mean),
                         pch = 19, col = roc_palette[[method]], cex = 0.65)
      }
    }
    x_ticks <- c(0, 0.05, 0.10, 0.15)
    y_ticks <- c(0, 0.5, 1)
    graphics::rect(bounds[["left"]], bounds[["bottom"]], bounds[["right"]], bounds[["top"]], border = "grey35")
    graphics::segments(map_x(x_ticks), bounds[["bottom"]], map_x(x_ticks), bounds[["bottom"]] - 0.008)
    graphics::segments(bounds[["left"]], map_y(y_ticks), bounds[["left"]] - 0.008, map_y(y_ticks))
    graphics::text(map_x(x_ticks), bounds[["bottom"]] - 0.027,
                   labels = c("0", ".05", ".10", ".15"), cex = 0.63)
    graphics::text(bounds[["left"]] - 0.018, map_y(y_ticks), labels = c("0", ".5", "1"), adj = 1, cex = 0.63)
    graphics::text(mean(bounds[c("left", "right")]), 0.565, "ID rejection <= 15%", cex = 0.70)
    return(invisible(NULL))
  }
  for (method in names(roc_palette)) {
    curve <- data$mean_curves[data$mean_curves$method == method, ]
    graphics::polygon(c(curve$id_false_rejection, rev(curve$id_false_rejection)),
                      c(curve$lower_sd, rev(curve$upper_sd)), border = NA,
                      col = grDevices::adjustcolor(roc_palette[[method]], alpha.f = 0.14))
    graphics::lines(curve$id_false_rejection, curve$ood_rejection_mean, col = roc_palette[[method]], lwd = 1.8)
    point <- data$nominal[data$nominal$method == method, ]
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
  graphics::par(mfrow = layout, mar = c(4, 4.2, 3.7, 0.7), oma = c(3.5, 0, 0.3, 0),
                mgp = c(2.5, 0.7, 0), las = 1, cex = 0.88)
  for (dataset in names(data)) {
    value <- data[[dataset]]
    title <- paste0(if (opts$allow_partial) "PARTIAL: " else "", helper$DATASETS[[dataset]],
                    " (n = ", length(value$metrics$states), ")")
    graphics::plot(NA_real_, NA_real_, xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i",
                   xlab = "ID false rejection rate", ylab = "OOD rejection rate", main = title)
    graphics::abline(0, 1, col = "grey75", lty = 3)
    draw_roc_curves(value)
    auc <- value$nominal$auroc_mean[match(names(roc_palette), value$nominal$method)]
    graphics::mtext(paste0("Mean AUROC: MaxSim ", sprintf("%.3f", auc[1]),
                           "; Discount ", sprintf("%.3f", auc[2]),
                           "; Energy ", sprintf("%.3f", auc[3])), side = 3, line = 0.4, cex = 0.70)
    draw_roc_curves(value, inset = TRUE)
  }
  graphics::mtext(unname(helper$METHODS[names(roc_palette)]), side = 1, outer = TRUE,
                  at = c(0.25, 0.5, 0.75), col = unname(roc_palette), line = 1, font = 2, cex = 0.95)
  graphics::mtext("Lines: mean ROC; shading: +/-1 SD across repetitions; points: mean calibrated operating points",
                  side = 1, outer = TRUE, line = 2.3, cex = 0.72)
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
  lines <- c("% Generated by experiments_ood/make_roc_plots.R from diagnostic exports.",
             "% Mean (standard error) across repetitions; rejection rates are fractions.",
             "\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
             helper$partial_banner(6L, paste(helper$DATASETS[opts$datasets], collapse = ", "), opts$allow_partial),
             "Dataset & Method & $\\alpha$ & AUROC & ID FR & OOD rejection \\\\", "\\midrule")
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
  helper <- roc_report_helpers(script)
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
  write_nominal_roc_table(nominal, opts, helper, path("ood_nominal_summary.tex"))
  metric_data <- lapply(data, `[[`, "metrics")
  helper$write_auroc_table(metric_data, opts, path("ood_summary_table.tex"))
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
    metric_helper_sha256 = helper$sha256_file(file.path(dirname(script), "make_plots.R")),
    roc_positive_class = "OOD", roc_detection_score = "negative ID score",
    averaging = "Per-repetition ROC with ordered vertical vertices retained; exact-FPR ties use the last vertex, interpolation proceeds to the first next larger-FPR vertex; arithmetic mean on a shared FPR grid",
    fpr_grid = grid, shading = "Mean +/- one standard deviation across repetitions, clipped to [0,1]; not a population confidence interval",
    auroc_summary = "Mean and standard error of individual repetition AUROCs; not the area of the interpolated mean curve",
    operating_points = "Separately averaged empirical rates at ID-calibration-selected nominal-alpha cutoffs; coordinates are not forced onto the mean ROC",
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
