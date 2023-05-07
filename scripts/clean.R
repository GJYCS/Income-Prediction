# Author: Jiayu Gu & Tianyi Li
# Title: MDML Final Project (Salary Prediction)
# Date: 05-11-2023

# Load library
library(tidyverse)
library(lubridate)
library(ROCR)
library(ranger)
library(lubridate)
library(dplyr)
library(haven)
library(wbstats)

# The data set contains samples of 3378291 rows and 21 columns of census data from 2005 to 2016
# The data set includes census data on only American and African immigrants
# Read data that was extracted from the IPUMS USA Website: https://usa.ipums.org/usa/index.shtml
# The code book is also avaiable in the page for each specific variables from the IPUMS USA website 

# Read data from the file
dat <- read_dta("replication_data.dta")

# Keep a copy of original data for convenience
census <- dat

# Check all names of the data
names(census)

# Keep 15 useful variables
# Unselect 'degfield', 'educ', colony', 'brcol', 'frcol', 'saharan'
census <- dat %>% 
  select(year, sex, age, marst, bpl, 
         bpld, yrimmig, yrsusa1, speakeng, educd, 
         citizen, wkswork2, uhrswork, incwage, lowgdp)

# Only keep observations between year 2006 to 2016
census <- census %>% 
  filter(year %in% c(2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016))

# Only keep observations between age 18 to 65
census <- census %>% 
  filter(age >= 18 & age <= 65)

# Drop if usual hours work per week is less than 25 hours or more than 80 hours
# Only keep respondents that worked full-time
census <- census %>%
  filter(uhrswork >= 25 & uhrswork <= 80)

# Drop unrealistic wages
# The national minimum hourly wage is 7.25/hr
# We keep respondents that work at least 25 hours a week for half year
# We drop rows with annual income less than 5000
# There are a large number of 999999 which seems unreliable, we decide to drop them all
census <- census %>%
  filter(incwage > 5000 & incwage < 999999)

# Remove all rows with NAs
census <- na.omit(census)

# Check empty rate to make sure there are no NAs
empty_rate <- census %>% summarise_all(list(name = ~sum(is.na(.))/length(.)))
empty_rate

# Re-code variable "educd" of highest educational attainment
census <- census %>% 
  mutate(educd = case_when(
    educd < 62 ~ "lessthanHS", 
    educd > 61 & educd < 65 ~ "HSgrad", 
    educd > 64 & educd < 101 ~ "somecollege", 
    educd == 101 ~ "Bachelor", 
    educd == 114 ~ "Master", 
    educd == 115 ~ "Professional", 
    educd == 116 ~ "Doctor"
    ))

# Re-code variable "speakeng" of English proficiency 
census <- census %>%
  mutate(speakeng = case_when(
    speakeng == 1 & speakeng == 6 ~ "limitEnglish",
    speakeng %in% c(2, 3, 4, 5) ~ "proficient"))

# Re-code variable of gender
census <- census %>%
  mutate(sex = case_when(
    sex == 1 ~ "Male",
    sex == 2 ~ "Female"))

# Re-code variable of citizenship status
census <- census %>%
  mutate(citizen = case_when(
    citizen %in% c(0, 1, 2) ~ "Citizen",
    citizen == 3 ~ "NotCitizen"))

# Re-code variable of Immigrant or not
census <- census %>%
  rename(birthplace = bpl) %>%
  mutate(birthplace = case_when(
    birthplace == 600 ~ "Africa",
    birthplace != 600 ~ "USA"))

# Re-code variable of Marital status
census <- census %>%
  rename(marital = marst) %>%
  mutate(marital = case_when(
    marital %in% c(1, 2, 3) ~ "Married",
    marital == 4 ~ "Divorced", 
    marital == 5 ~ "Widowed",
    marital == 6 ~ "Single"))

# Re-code variable of weeks worked last year
census <- census %>%
  mutate(wkswork2 = case_when(
    wkswork2 == 1 ~ "1-13 weeks",
    wkswork2 == 2 ~ "14-26 weeks",
    wkswork2 == 3 ~ "27-39 weeks",
    wkswork2 == 4 ~ "40-47 weeks",
    wkswork2 == 5 ~ "48-49 weeks",
    wkswork2 == 6 ~ "50-52 weeks"))

# Drop bpld & lowgdp variable as they are not useful for modeling
census <- census %>%
  select(-bpld, -lowgdp)

# Remove all NA again after we re-define the columns
census <- na.omit(census)

# Check empty rate again
empty_rate

# Next, adjust annual income with inflation 
# Use wb_data from "wbstats" package to access real-world economic indicators
# We extract data on US annual inflation rates from 2006 to 2016 to calculate adjusted real income
# Use inflation / 100 to convert a percentage to decimal
inflation_rates <- wb_data("FP.CPI.TOTL.ZG", country = "US", start_date = 2006, end_date = 2016)
inflation_rates <- inflation_rates %>%
  rename(year = date,
         inflation = FP.CPI.TOTL.ZG) %>%
  mutate(inflation = inflation / 100) %>%
  select(year, inflation)

# Add up all inflation rates which increased by year
inflation_rates <- inflation_rates %>% 
  mutate(cumulative_inflation_rate_by_year = 
           sapply(inflation, function(x) {
             index <- which(inflation == x)
             cumprod(1 + inflation[index:length(inflation)])
           }))

# Use 2016 as the base
# Set 2016's cumulative inflation rate as 1
inflation_rates$cumulative_inflation_rate_by_year[nrow(inflation_rates)] <- 1

# Find each year's cumulative inflation rate
# The second-to-last position in each list will be the rate for that year relative to 2016
inflation_rates <- inflation_rates %>% 
  mutate(cumulative_inflation_rate = sapply(cumulative_inflation_rate_by_year, 
                                            tail, n = 2)) %>% 
  mutate(cumulative_inflation_rate = sapply(cumulative_inflation_rate, 
                                            head, n = 1)) %>%
  select(-inflation) %>% 
  select(-cumulative_inflation_rate_by_year)

# Merge the inflation data with census data
# Calculate the adjusted income
merged_data <- merge(census, inflation_rates, by = "year")
merged_data <- merged_data %>%
  mutate(adjusted_income = incwage * cumulative_inflation_rate)

# Determine whether the adjusted real income is above national median in 2016 (35K USD)
# Information from FRED Economic Data
# Source from U.S. Census Bureau
merged_data <- merged_data %>%
  mutate(outcome = ifelse(adjusted_income >= 35000, 1, 0))
