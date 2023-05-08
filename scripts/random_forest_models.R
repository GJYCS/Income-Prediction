##################################################
###
### Author: Jiayu Gu & Tianyi Li
### Title: MDML Final Project (Salary Prediction)
### Purpose: Fit random tree models
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


m1 <- ranger(outcome ~ sex + age + marital + educd + yrinus + wkswork2 + uhrswork, 
             num.tree = 100, 
             data = train, 
             probability = TRUE)

# AUC
test <- test %>%
  mutate(rf.predicted.probability = predict(m1, test)$predictions[,2])
test.rf.pred <- prediction(test$rf.predicted.probability, test$outcome)
test.rf.perf <- performance(test.rf.pred, "auc")
cat('The auc score for test is', test.rf.perf@y.values[[1]], "\n")


confusionMatrix
