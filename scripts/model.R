##################################################
###
### Author: Jiayu Gu & Tianyi Li
### Title: MDML Final Project (Salary Prediction)
### Purpose: Fit models (GLM & GBM & Random Forest) 
### Date: 05-11-2023
### Input: census_cleaned.csv
### Output: figures
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
library(MASS)
library(gbm)
library(pROC)

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

##################################################

# Firstly, we want to split the train sets and the test sets
# We have two different methods:

# Method 1
# Randomly shuffle census and split it in half, storing the results as train and test sets
# shuffle census
census1 <- census[sample(nrow(census)), ]
# split in half
census_half <- nrow(census1) / 2
# Train set 2: The first half
train1 <- census1[1:census_half, ]
# Test set 2: The second half
test1 <- census1[(nrow(train1)+1) : nrow(census1), ]

# Method 2
# Randomly split census into an 80% train set and a 20% test set
# Each person should be in either the training or testing set 
# No person should be divided across the two sets
# shuffle census
census2 <- census[sample(nrow(census)), ]
# Train set 3: 80% of the data
train2 <- census2 %>% 
  slice(1 : floor(nrow(census2)*0.8))
# Test set 3: 20% of the data
test2 <- census2 %>% 
  slice(floor(nrow(census2)*0.8)+1 : n())
##################################################

# Secondly, we start to model. We will build three different types of models for income predictions
# Included: GLM (generalized linear model), GBM (Gradient Boosting Machine model), and Random Forest model

###Using sampling method 1 as example###

###GLM example###
m1 <- glm(outcome ~ sex + age + marital + educd + wkswork2 + uhrswork + birthplace, 
          data = train1, family = "binomial")
summary(m1)

# GLM AUC
test1 <- test1 %>%
  mutate(logistic.predicted.probability = predict(m1, newdata = test1, type='response'))
test1.logistic.pred <- prediction(test1$logistic.predicted.probability, test1$outcome)
test1.logistic.perf <- performance(test1.logistic.pred, "auc")
cat('The auc score for test is', test1.logistic.perf@y.values[[1]], "\n")

# check multicollinearity
vif(m1)
# All VIF less than 2.5 indicates low correlation of that predictor with other predictors.
# No multicollinearity problem

###GBM example###
# re-factor for GBM fitting
train11 <- train1 %>% 
  mutate(year = as.factor(year), 
         sex = as.factor(sex), 
         marital = as.factor(marital), 
         birthplace = as.factor(birthplace), 
         speakeng = as.factor(speakeng), 
         educd = as.factor(educd), 
         citizen = as.factor(citizen),
         wkswork2 = as.factor(wkswork2)
  )

test11 <- test1 %>% 
  mutate(year = as.factor(year), 
         sex = as.factor(sex), 
         marital = as.factor(marital), 
         birthplace = as.factor(birthplace), 
         speakeng = as.factor(speakeng), 
         educd = as.factor(educd), 
         citizen = as.factor(citizen),
         wkswork2 = as.factor(wkswork2)
  )

# Fit GBM model
m2 <- gbm(outcome ~ sex + age + marital + speakeng + educd + citizen + wkswork2 + uhrswork + yrinus, 
          data = train11, 
          distribution = "bernoulli", 
          n.trees = 100, 
          interaction.depth = 3)

# Print GBM model summary
print(m2)
summary(m2)

# Calculate GBM's AUC score
# Make predictions on test data
test11$gbm.predicted.probability <- predict(m2, newdata = test11, n.trees = 100, type = "response")
test11.gbm.pred <- prediction(test11$gbm.predicted.probability, test11$outcome)
test11.gbm.perf <- performance(test11.gbm.pred, "auc")
cat('The AUC score for test is', test11.gbm.perf@y.values[[1]], "\n")

###Random Forest example###
m3 <- ranger(outcome ~ sex + age + marital + speakeng + educd + citizen + wkswork2 + uhrswork + yrinus, 
             num.tree = 100, 
             data = train1, 
             probability = TRUE)

# AUC
test1 <- test1 %>%
  mutate(rf.predicted.probability = predict(m3, test1)$predictions[,2])

test1.rf.pred <- prediction(test1$rf.predicted.probability, test1$outcome)
test1.rf.perf <- performance(test1.rf.pred, "auc")
cat('The auc score for test is', test1.rf.perf@y.values[[1]], "\n")


##################################################

# Thirdly, we will create some plots for visualization and comparations
# Included: Calibration, precision, recall-at-k%
# 6.make calibration plot
plot.data <- test %>% mutate(calibration = round(100*logistic.predicted.probability)) %>%
  group_by(calibration) %>% summarize(model.estimate = mean(logistic.predicted.probability),
                                      numstops = n(),
                                      empirical.estimate = mean(outcome))

# create and save plot
p <- ggplot(data = plot.data, aes(y=empirical.estimate, x=model.estimate))
p <- p + geom_point(alpha=0.5, aes(size=numstops))
p <- p + scale_size_area(guide='none', max_size=15)
p <- p + geom_abline(intercept=0, slope=1, linetype="dashed")
p <- p + scale_y_log10('Empirical probability \n', limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p <- p + scale_x_log10('\nModel estimated probability', limits=c(.001,1), 
                       breaks=c(.001,.003,.01,.03,.1,.3,1),
                       labels=c('0.1%','0.3%','1%','3%','10%','30%','100%'))
p


# precision
# Logistic regression model
logistic_data <- test %>%
  # rank from high to low
  arrange(desc(logistic.predicted.probability)) %>%
  # calculate precision
  mutate(outcome = as.numeric(outcome),
         rank = row_number(),
         precision = cumsum(as.numeric(as.character(outcome))[rank]) / rank) %>%
  mutate(model = "Logistic Regression") %>% 
  select(rank, precision, model)

p <- ggplot(data=logistic_data, aes(x=rank, y=precision, 
                                    group=factor(model), 
                                    color=factor(model)))
p <- p + geom_line() 
p <- p + scale_x_continuous("Number of restaurants", 
                            limits=c(100, nrow(test)), 
                            labels=scales::comma) 

p <- p + scale_y_continuous("Precision") + labs(color = "Model")
p

# confusionMatrix

