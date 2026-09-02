if (!requireNamespace("DoublyRobustHD", quietly = TRUE)) {
  stop("Install DoublyRobustHD before regenerating Bayes-DR reference fixtures")
}

reference_sha <- "617945098b2f540039367d4398068ba9ae713891"
installed_sha <- utils::packageDescription("DoublyRobustHD")$RemoteSha
if (!is.null(installed_sha) && installed_sha != reference_sha) {
  warning("DoublyRobustHD was installed from a different Git commit")
}

output_dir <- "test/reference"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(1401)
n_binary <- 30
x_binary <- matrix(rnorm(n_binary * 3), n_binary, 3)
t_binary <- rbinom(
  n_binary,
  1,
  pnorm(0.8 * x_binary[, 1] - 0.5 * x_binary[, 2])
)
y_binary <- 0.7 * t_binary + 0.6 * x_binary[, 1] +
  0.3 * x_binary[, 3] + rnorm(n_binary, sd = 0.5)
binary_data <- data.frame(
  y = y_binary,
  treatment = t_binary,
  x1 = x_binary[, 1],
  x2 = x_binary[, 2],
  x3 = x_binary[, 3]
)
write.csv(
  binary_data,
  file.path(output_dir, "bayes_dr_binary.csv"),
  row.names = FALSE
)

set.seed(2401)
binary_fit <- DoublyRobustHD::DRbayes(
  y = y_binary,
  t = t_binary,
  x = x_binary,
  nScans = 1000,
  nBurn = 500,
  thin = 2,
  nBoot = 500,
  lower = 0.01,
  upper = 0.99
)

set.seed(1402)
n_continuous <- 24
x_continuous <- matrix(rnorm(n_continuous * 2), n_continuous, 2)
t_continuous <- 0.7 * x_continuous[, 1] -
  0.4 * x_continuous[, 2] + rnorm(n_continuous)
y_continuous <- 1 + 0.6 * t_continuous - 0.1 * t_continuous^2 +
  0.5 * x_continuous[, 1] + rnorm(n_continuous, sd = 0.4)
continuous_data <- data.frame(
  y = y_continuous,
  treatment = t_continuous,
  x1 = x_continuous[, 1],
  x2 = x_continuous[, 2]
)
write.csv(
  continuous_data,
  file.path(output_dir, "bayes_dr_continuous.csv"),
  row.names = FALSE
)

grid <- c(-0.5, 0, 0.5)
set.seed(2402)
continuous_fit <- DoublyRobustHD::DRbayesER(
  y = y_continuous,
  t = t_continuous,
  x = x_continuous,
  locations = grid,
  nScans = 1000,
  nBurn = 500,
  thin = 2,
  nBoot = 500,
  threshold = 1e-5
)

expected <- rbind(
  data.frame(
    scenario = "binary",
    location = NA_real_,
    estimate = binary_fit$TreatEffect,
    standard_error = binary_fit$TreatEffectSE,
    lower = binary_fit$TreatEffectCI[1],
    upper = binary_fit$TreatEffectCI[2]
  ),
  data.frame(
    scenario = "continuous",
    location = grid,
    estimate = as.numeric(continuous_fit$TreatEffect),
    standard_error = as.numeric(continuous_fit$TreatEffectSE),
    lower = as.numeric(continuous_fit$TreatEffectCI[, 1]),
    upper = as.numeric(continuous_fit$TreatEffectCI[, 2])
  )
)
write.csv(
  expected,
  file.path(output_dir, "bayes_dr_expected.csv"),
  row.names = FALSE,
  na = ""
)
