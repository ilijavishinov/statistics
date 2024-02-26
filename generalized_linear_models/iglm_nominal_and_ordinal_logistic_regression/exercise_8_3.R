require(MASS)

data = read.csv("C:\\Users\\ilija\\OneDrive\\Desktop\\rm\\exercises\\chapter_8\\data_8_3_flat_processed.csv")

# ________________________________________________________________________________________
# a) proportional odds model
data$stage = as.factor(data$stage)
# model using no change as reference category
model = polr(relevel(stage, ref = "1") ~ treatment + sex, data = data, weights = frequency, Hess = TRUE)
# reordering columns
data = data[, c('stage', 'treatment', 'sex', 'frequency')]
summary(model)

# ________________________________________________________________________________________
# b) adequacy of model
grouped_frequencies = c(
  28+45+29+26,
  4+12+5+2,
  41+44+20+20,
  12+7+3+1
)
predictions = predict(model, data, type = 'probs')
# reordering predictions columns so that we have 0, 1, 2, 3 for encoded "stage"
predictions = cbind(predictions[, 2],
                    predictions[, 1],
                    predictions[, 3],
                    predictions[, 4])

# selecting prediction rows since there are duplicates for each "stage" category
predictions = rbind(predictions[1,],
                    predictions[5,],
                    predictions[9,],
                    predictions[13,])
# expected frequencies
expected_freq_mat = grouped_frequencies * predictions
expected_freq_flat = as.vector(t(expected_freq_mat))
data$expected_frequency = expected_freq_flat

# pearson residuals and goodness of fit statistic
data$pearson_res = (data$frequency - data$expected_frequency) / sqrt(data$expected_frequency)
chi_square = sum(data$pearson_res^2)
# chi_square = 13.79548
p_value = pchisq(chi_square, df=7, lower.tail=FALSE)
# p_value = 0.05494089

# the model is not accepted at 95% significance

# ________________________________________________________________________________________
# c) Wald statistic
(coeftable <- coef(summary(model)))
wald_treatment = coeftable['treatment', 'Value'] / coeftable['treatment', 'Std. Error']
# W = 0.7021018
wald_p_value = pnorm(wald_treatment, lower.tail=FALSE)
# p_value = 0.4826157

# wald statistic shows no evidence for treatment difference

# ________________________________________________________________________________________
# d) two proportional odds models to test the hypothesis of no treatment difference
model_without_treatment = polr(relevel(stage, ref = "1") ~ sex, data = data, weights = frequency, Hess = TRUE)
summary(model_without_treatment)
deviance_statistic = model_without_treatment$deviance - model$deviance
# D = 0.4931418
deviance_p_value = pchisq(deviance_statistic, df=1, lower.tail=FALSE)
# p_value = 0.4825292

# deviance statistic showss no evidence for treatment difference
