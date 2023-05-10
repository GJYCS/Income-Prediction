##################################################
###
### Author: Jiayu Gu & Tianyi Li
### Title: MDML Final Project (Salary Prediction)
### Purpose: Fit models (GLM & GBM & Random Forest) 
### Date: 05-11-2023
### Input: census_cleaned.csv
### Output: 
### calibration_glm.png
### calibration_gbm.png
### calibration_rf.png
### performance_plot_all.png
###
##################################################

# Load library
library(tidyverse)
library(lubridate)
library(ROCR)
library(ranger)
library(lubridate)
library(dplyr)
library(haven)
library(wbstats)
library(car)
library(gbm)
library(rms)
library(caret)

# Set the seed to 47196
set.seed(47196)

# Load data
census <- read_csv("~/Desktop/2047 Messy Data and Machine Learning/PROJECT/data/census_cleaned.csv")

# Find variables in census data
names(census)

# In this cleaned census dataset, we have 949720 observations of 13 variables
# 13 variables include: year, sex, age, marital, birthplace, speakeng, educd, citizen, wkswork2, uhrswork, yrinus, adjusted_income, outcome
# Among these 13 variables, there is 1 dependent variable: outcome, and 12 independent variables
# During data cleaning process, we factor 9 variables include: year, sex, marital, birthplace, speakeng, educd, citizen, wkswork2, and outcome
# The other 4 variables are continuous: age, uhrswork, yrinus, and adjusted_income

# By restating our research question on: 
# "How is one’s educational attainment and other demographic characteristics associated with his earning performance?"
# We decided to drop the variable 'year' since we are not aiming to see the change by year
# Also, we do not want to fit 'adjusted_income' in our model because it has been re-defined as outcome.
census <- census %>% 
  select(-year, -adjusted_income) 
# Now, we have 11 variables. 
# Factor variables
census <- census %>% 
  mutate(sex = as.factor(sex), 
         marital = as.factor(marital), 
         birthplace = as.factor(birthplace), 
         speakeng = as.factor(speakeng), 
         educd = as.factor(educd), 
         citizen = as.factor(citizen),
         wkswork2 = as.factor(wkswork2), 
         outcome = as.factor(outcome))

##################################################

# First, we want to split the train sets and the test sets
# We will randomly split census into an 80% train set and a 20% test set (And use the same method for further sampling)
# Each person should be in either the training or testing set 
# No person should be divided across the two sets

# shuffle census
census <- census[sample(nrow(census)), ]
# Train set 3: 80% of the data
train <- census %>% 
  slice(1 : floor(nrow(census)*0.8))
# Test set 3: 20% of the data
test <- census %>% 
  slice(floor(nrow(census)*0.8)+1 : n())

# Train set has 759776 obs. of 11 variables.
# Test set has 189944 obs. of 11 variables.
# The size of our cleaned dataset 'census' is large enough
# We believe the way of splitting train/test set will not cause much difference in the results
# Thus, one split method is enough for all models use
# We can also do a 50%, 50% split, but we believe a large size of train set could enhance our models

##################################################

# Second, we start to model. 
# We will build three different types of models for income predictions.

# Included: 
# MODEL:
# GLM (Logistic Regression model), GBM (Gradient Boosting Model), and Random Forest Model
# VARIABLES:
# sex: female, male
# age: 18 to 65
# marital: marriage status
# birthplace: the location where a person was born
# speakeng: one's English speaking level
# educd: one's educational attainment
# citizen: whether someone is a citizen of the United States
# yrinus: the length of time someone has lived in the United States
# outcome: whether one's adjusted real income is above national median in 2016 (35K USD)

# Some independent variables may seem overlap (such as birthplace, citizen, and yrinus), but 
# Each variable measures a distinct aspect of an individual's background and experience
# These variables could potentially have different effects on our dependent variable, and therefore may all be relevant to include in our model
# We will keep all of them and test for multicollinearity after modeling

#######################GLM########################

# Fit a glm model
glm <- glm(outcome ~ sex + age + marital + birthplace + speakeng + 
             educd + citizen + wkswork2 + uhrswork + yrinus, 
          data = train, family = "binomial")

summary(glm)
# Since 'age' has a p-value of 0.312 > 0.5, we cannot conclude an effect of age on outcome. 
# Other variables all have a p-value <2e-16, thus we can conclude their effects on our outcome.

# Check multicollinearity
vif(glm)
# Since 'age' has a p-value of 0.312 > 0.5, we cannot conclude an effect of age on outcome. 
# According to the vif redult, 'age' has a GVIF of 21.3; 'birthplace' has a GVIF of 6.78, and 'yrinus' has a GVIF of 27.23316
# Indicating 'age', 'birthplace', and 'yrinus' show muticollinearity, that makes sense
# We decide to drop 'yrinus' first, and take a look at new glm model.


# Fit a new glm model
glm2 <- glm(outcome ~ sex + age + marital + birthplace + speakeng + 
              educd + citizen + wkswork2 + uhrswork,
           data = train, family = "binomial")

summary(glm2)
# All variables have a p-value <2e-16, thus we can conclude their effects on our outcome.

# Check multicollinearity again
vif(glm2)
# No multicollinearity issue, all below 2

# GLM2 AUC
test <- test %>%
  mutate(logistic.predicted.probability = predict(glm2, newdata = test, type='response'))
test.logistic.pred <- prediction(test$logistic.predicted.probability, test$outcome)
test.logistic.perf <- performance(test.logistic.pred, "auc")
cat('The auc score for test is', test.logistic.perf@y.values[[1]], "\n")
# The auc score for test is 0.8095131.
# Task complete

#######################GBM########################

# When we are fitting GLM model, we find 'yrinus' show severe multicollinearity
# Multicollinearity is a property of the data rather than the model
# It is likely that 'yrinus' will also exhibit multicollinearity in other models (GBM model & RF model because they are both tree-based)
# Thus, we choose not include this variable in other models as well

# re-factor outcome for gbm model use
train1 <- train %>% 
  mutate(outcome = as.numeric(outcome) - 1)

test1 <- test %>% 
  mutate(outcome = as.numeric(outcome) - 1)

gbm <- gbm(outcome ~ sex + age + marital + birthplace + speakeng + 
             educd + citizen + wkswork2 + uhrswork, 
          data = train1, 
          distribution = "bernoulli", 
          n.trees = 100, 
          interaction.depth = 3)

# Print GBM model summary
print(gbm)
# There were 9 predictors of which 9 had non-zero influence.
summary(gbm)

# Calculate GBM's AUC score
# Make predictions on test1 data with n.trees = 100
test$boosting.predicted.probability <- predict(gbm, newdata = test1, n.trees = 100, type = "response")
test.boosting.pred <- prediction(test$boosting.predicted.probability, test$outcome)
test.boosting.perf <- performance(test.boosting.pred, "auc")
cat('The AUC score for test is', test.boosting.perf@y.values[[1]], "\n")
# The AUC score for test is 0.8275674.
# Task complete

################Random Forest####################

# Fit the random forest model
rf <- ranger(outcome ~ sex + age + marital + birthplace + speakeng + 
               educd + citizen + wkswork2 + uhrswork, 
             num.tree = 100, 
             data = train, 
             probability = TRUE)

# Calculate rf's AUC score
test <- test %>%
  mutate(rf.predicted.probability = predict(rf, test)$predictions[,2])
test.rf.pred <- prediction(test$rf.predicted.probability, test$outcome)
test.rf.perf <- performance(test.rf.pred, "auc")
cat('The auc score for test is', test.rf.perf@y.values[[1]], "\n")
# The auc score for test is 0.8285135.
# Task complete

# Use confusionMatrix to compare all three models

# GLM
confusionMatrix(as.factor(ifelse(test$logistic.predicted.probability>= 0.5, 1, 0)), test$outcome)
# Accuracy: 0.756 Specificity : 0.8550

# GBM
confusionMatrix(as.factor(ifelse(test$boosting.predicted.probability>= 0.5, 1, 0)), test$outcome)
# Accuracy : 0.7677 Specificity : 0.8750

# RF*
confusionMatrix(as.factor(ifelse(test$rf.predicted.probability>= 0.5, 1, 0)), test$outcome)
# Accuracy : 0.7685 Specificity : 0.8762

##################################################

# Thirdly, we will create some plots for visualization and comparations
# Included: 
# For each model: one calibration plot 
# For all three models: one plot with three precision-at-k curves

# Three ggplot2() default color: 
# The hex color code for the red in the plot is #F8766D.
# The hex color code for the green in the plot is #00BA38.
# The hex color code for the blue in the plot is #619CFF.

#####################GLM##########################

# make a calibration plot
glm.c.plot.data <- test %>% mutate(glm.calibration = round(100*logistic.predicted.probability)) %>%
  group_by(glm.calibration) %>% summarise(glm.estimate = mean(logistic.predicted.probability),
                                          numstops = n(),
                                          empirical.estimate = mean(as.numeric(outcome)-1))

# create and save plot
p <- ggplot(data = glm.c.plot.data, aes(y=empirical.estimate, x=glm.estimate))
p <- p + geom_point(alpha=0.5, aes(size=numstops), color = "#00BA38")
p <- p + scale_size_area(guide='none', max_size=15)
p <- p + geom_abline(intercept=0, slope=1, linetype="dashed")
p <- p + scale_y_log10('Empirical probability \n', limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p <- p + scale_x_log10('\nLogistic Regression Model estimated probability', 
                       limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p 

#ggsave(plot=p, file="~/Desktop/2047 Messy Data and Machine Learning/PROJECT/figures/calibration_glm.png")

#####################GBM##########################

# make a calibration plot
gbm.c.plot.data <- test %>% mutate(gbm.calibration = round(100*boosting.predicted.probability)) %>%
  group_by(gbm.calibration) %>% summarise(gbm.estimate = mean(boosting.predicted.probability),
                                          numstops = n(),
                                          empirical.estimate = mean(as.numeric(outcome)-1))

# create and save plot
p <- ggplot(data = gbm.c.plot.data, aes(y=empirical.estimate, x=gbm.estimate))
p <- p + geom_point(alpha=0.5, aes(size=numstops), color = "#F8766D")
p <- p + scale_size_area(guide='none', max_size=15)
p <- p + geom_abline(intercept=0, slope=1, linetype="dashed")
p <- p + scale_y_log10('Empirical probability \n', limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p <- p + scale_x_log10('\nGradient Boosting Model estimated probability', 
                       limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p

#ggsave(plot=p, file="~/Desktop/2047 Messy Data and Machine Learning/PROJECT/figures/calibration_gbm.png")

##################RandomForest#####################

# make a calibration plot
rf.c.plot.data <- test %>% mutate(rf.calibration = round(100*rf.predicted.probability)) %>%
  group_by(rf.calibration) %>% summarise(rf.estimate = mean(rf.predicted.probability),
                                         numstops = n(),
                                         empirical.estimate = mean(as.numeric(outcome)-1))

# create and save plot
p <- ggplot(data = rf.c.plot.data, aes(y=empirical.estimate, x=rf.estimate))
p <- p + geom_point(alpha=0.5, aes(size=numstops), color = "#619CFF")
p <- p + scale_size_area(guide='none', max_size=15)
p <- p + geom_abline(intercept=0, slope=1, linetype="dashed")
p <- p + scale_y_log10('Empirical probability \n', limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p <- p + scale_x_log10('\nRandom Forest Model estimated probability',
                       limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p

#ggsave(plot=p, file="~/Desktop/2047 Messy Data and Machine Learning/PROJECT/figures/calibration_rf.png")

####################3 MODELS######################

# Logistic regression model
logistic_data <- test %>%
  # rank from high to low
  arrange(desc(logistic.predicted.probability)) %>%
  # calculate precision
  mutate(outcome = as.numeric(outcome) - 1,
         rank = row_number(),
         precision = cumsum(as.numeric(as.character(outcome))[rank]) / rank) %>%
  mutate(model = "Logistic Regression") %>% 
  select(rank, precision, model)

# Gradient Boosting Model
gbm_data <- test %>%
  # rank from high to low
  arrange(desc(boosting.predicted.probability)) %>%
  # calculate precision
  mutate(outcome = as.numeric(outcome) - 1,
         rank = row_number(),
         precision = cumsum(as.numeric(as.character(outcome))[rank]) / rank) %>%
  mutate(model = "Gradient Boosting") %>% 
  select(rank, precision, model)

# Random forest model
rf_data <- test %>%
  # rank from high to low
  arrange(desc(rf.predicted.probability)) %>%
  # calculate precision
  mutate(outcome = as.numeric(outcome) - 1,
         rank = row_number(),
         precision = cumsum(as.numeric(as.character(outcome))[rank]) / rank) %>%
  mutate(model = "Random Forest") %>% 
  select(rank, precision, model)

# combine data
plot.data <- bind_rows(logistic_data, gbm_data, rf_data)

# make plot
theme_set(theme_bw())
p <- ggplot(data=plot.data, aes(x=rank, y=precision, 
                                group=factor(model), 
                                color=factor(model)))
p <- p + geom_line() 
# start with 100 to make more sense
# Number of Observations: 189944
p <- p + scale_x_log10("Number of Observations", 
                       limits=c(100, nrow(test)),
                       breaks=c(100, 500, 1000, 5000, 10000, 50000, 100000), 
                       labels=c("100", "500", "1,000", "5,000", "10,000", "50,000", "100,000"))
p <- p + scale_y_continuous("Precision", limits=c(0, 1), labels=scales::percent)
p <- p + labs(color = "Model")

# show the plot
p

# save the plot
#ggsave(plot=p, file="~/Desktop/2047 Messy Data and Machine Learning/PROJECT/figures/performance_plot_all.png")
