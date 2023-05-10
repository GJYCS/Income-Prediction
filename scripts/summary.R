##################################################
###
### Author: Jiayu Gu & Tianyi Li
### Title: MDML Final Project (Salary Prediction)
### Purpose: Statistics Summary  
### Date: 05-11-2023
### Input: census_cleaned.csv
### Output: 
###
##################################################

# Load data
census <- read_csv("~/Desktop/2047 Messy Data and Machine Learning/PROJECT/data/census_cleaned.csv")


##### Statistics summary


### Education 

# count the number of observations in each education category
education_counts <- table(census$educd)

# calculate the percentage of each category
education_pct <- prop.table(education_counts) * 100

# print the result
print(education_pct)

# Calculate mean and median income for each education category
income_stats <- aggregate(adjusted_income ~ educd, data = census, FUN = function(x) c(mean = mean(x), median = median(x)))

# Print the result
print(income_stats)

# Calculate the counts of each education category with the income outcome
outcome_education_counts <- table(census$educd, census$outcome)

# Calculate the percentages of "1" within each education category
outcome_education_percentages <- prop.table(outcome_education_counts, margin = 1) * 100

# Print the percentages
print(outcome_education_percentages)


### Gender

sex_counts <- table(census$sex)
sex_pct <- prop.table(sex_counts)
print(sex_pct)
aggregate(adjusted_income ~ sex, data = census, FUN = function(x) c(mean = mean(x), median = median(x)))
prop.table(table(census$sex, census$outcome), margin = 1) * 100


### Citizenship
citizen_counts <- table(census$citizen)
citizen_pct <- prop.table(citizen_counts)
print(citizen_pct)
aggregate(adjusted_income ~ citizen, data = census, FUN = function(x) c(mean = mean(x), median = median(x)))
prop.table(table(census$citizen, census$outcome), margin = 1) * 100


### Marriage status
marriage_counts <- table(census$marital)
marriage_pct <- prop.table(marriage_counts)
print(marriage_pct)
aggregate(adjusted_income ~ marital, data = census, FUN = function(x) c(mean = mean(x), median = median(x)))
prop.table(table(census$marital, census$outcome), margin = 1) * 100

