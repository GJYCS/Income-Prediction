##################################################
###
### Author: Jiayu Gu & Tianyi Li
### Title: MDML Final Project (Salary Prediction)
### Purpose: Fit GLMs (generalized linear models) 
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

# Set the seed to 2048
set.seed(2048)

# Load data
census <- read_csv("./data/census_cleaned.csv")

# Find variables in census data
names(census)

# In this cleaned census dataset, we have 1039919 observations of 14 variables.
# 14 variables include: year, sex, age, marital, birthplace, speakeng, educd, citizen, wkswork2, uhrswork, incwage, yrinus, adjusted_income, outcome
# Among these 14 variables, there is 1 dependent variable: outcome, and 13 independent variables.
# During data cleaning process, we factor 9 variables include: year, sex, marital, birthplace, speakeng, educd, citizen, wkswork2, and outcome
# The other 5 variables are continuous: age, uhrswork, incwage, yrinus, and adjusted_income

##################################################
# First, we want to split the train sets and the test sets
# We have three methods:

# Method 1
# We split the data into two pieces: the first five years and the second five years
# Train set 1: From year 2007 to 2011
train1 <- census %>% 
  filter(year %in% c(2007, 2008, 2009, 2010, 2011)) 
# Test set 1: From year 2012 to 2016
test1 <- census %>% 
  filter(year %in% c(2012, 2013, 2014, 2015, 2016)) 

# Method 2
# Randomly shuffle census and split it in half, storing the results as train and test sets
# shuffle census
census2 <- census[sample(nrow(census)), ]
# split in half
census_half <- nrow(census2) / 2
# Train set 2: The first half
train2 <- census2[1:census_half, ]
# Test set 2: The second half
test2 <- census2[(nrow(train2)+1) : nrow(census2), ]

# Method 3
# Randomly split census into an 80% train set and a 20% test set
# so each person should be in either the training or testing set; no person should be divided across the two sets
# shuffle census
census3 <- census[sample(nrow(census)), ]
# Train set 3: 80% of the data
train3 <- census3 %>% 
  slice(1 : floor(nrow(census3)*0.8))
# Test set 3: 20% of the data
test3 <- census3 %>% 
  slice(floor(nrow(census3)*0.8)+1 : n())
##################################################



m1 <- glm(outcome ~ sex + age + marital + educd + wkswork2 + uhrswork, 
          data = train, family = "binomial")
summary(m1)

# AUC
test <- test %>%
  mutate(logistic.predicted.probability = predict(m1, newdata = test, type='response'))
test.logistic.pred <- prediction(test$logistic.predicted.probability, test$outcome)
test.logistic.perf <- performance(test.logistic.pred, "auc")
cat('The auc score for test is', test.logistic.perf@y.values[[1]], "\n")

# check multicollinearity
library(car)
vif(m1)
# All VIF less than 1.31 indicates low correlation of that predictor with other predictors.
# No multicollinearity problem


# Make recall-at-k% plot
plot.data <- test %>% 
  arrange(desc(logistic.predicted.probability) ) %>%
  mutate(numstops = row_number(), 
         percent.outcome = cumsum(outcome)/sum(outcome),
         stops = numstops/n()) %>% select(stops, percent.outcome)

theme_set(theme_bw())
p <- ggplot(data=plot.data, aes(x=stops, y=percent.outcome))
p <- p + geom_line()
p <- p + scale_x_log10('\nPercent of people', limits=c(0.003, 1), breaks=c(.003,.01,.03,.1,.3,1),
                       labels=c('0.3%','1%','3%','10%','30%','100%'))
p <- p + scale_y_continuous("Percent of Adjusted Real Income Above 35k", limits=c(0, 1), labels=scales::percent)
p





