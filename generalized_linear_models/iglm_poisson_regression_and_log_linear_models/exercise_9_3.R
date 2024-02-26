require(MASS)

data <- read.csv("C:\\Users\\ilija\\OneDrive\\Desktop\\rm\\exercises\\chapter_9\\data_9_3_processed.csv")
head(data)
summary(data)

# ________________________________________________________________________________________
# a) part 2: log-linear model
# log-linear model
model = glm(freq ~ vaccine + levels,family=poisson, data=data)
data$residuals = resid(model, type = 'pearson')

# chi-square statistic
chi_square = sum(data$residuals^2)
# chi_square = 17.94204
p_value = pchisq(chi_square, df=6-1-2, lower.tail=FALSE)
# p_value = 0.000452124

# we conclude that the distribution is not the same for the
# placebo and vaccine groups

# ________________________________________________________________________________________
# b) homogeneity of response distribution model
# I found that the estimates of the homogeneity of response distribution model
# are the same as the expected frequencies of the conventional chi square test

# coppied from the conventional chi-square solution in exercise_9_3.ipynb
data$conventional_expected = c(16.136986,
                               14.863014,
                               13.534247,
                               12.465753,
                               8.328767,
                               7.671233)
# pearson residuals
data$conv_pearson_res = (data$freq - data$conventional_expected) / sqrt(data$conventional_expected)
# chi-square statistic
chi_square = sum(data$conv_pearson_res^2)
# chi_square = 17.64783
p_value = pchisq(chi_square, df=6-1-2, lower.tail=FALSE)
# dof = 3, because there are 2 paramteres in the model
# p_value = 0.0005198776

# we conclude that the distribution is not the same for the
# placebo and vaccine groups

# ________________________________________________________________________________________
# c) ordinal logistic regression and cutpoints
ordinal_log_reg = polr(factor(levels) ~ vaccine, data = data, weights = freq, Hess = TRUE)
predictions = predict(ordinal_log_reg, data, type = 'probs')
cut_points_placebo = cumsum(predictions[1,])
# C1 and C2 for placebo group are 0.6376976 0.9199318
cut_points_vaccine = cumsum(predictions[2,])
# C1 and C2 for vaccine group are 0.2188492 0.6464910

