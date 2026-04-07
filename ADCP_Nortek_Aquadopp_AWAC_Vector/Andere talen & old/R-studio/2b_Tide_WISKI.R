#******************************************************************************
# Name:        Tide_statistics
# Purpose:     Import WISKI7 timeseries and generate tide statistics. The outputs are used in scripts for e.g. the yearly Moneos reports
# Author:      Melanka Brackx - adapted Dieter Meire (WL)
#
# Created:     2019-2020
#******************************************************************************
# method:
# Calculation of yearly average tide statistics:
# \itemize{
# \item levels: water level statistics
# \item lags: time lag with regard to a reference station.
# \item durations: Durations of rising and falling
# }
#
# The script relies fully on the HICtools package for the calculation of the statistics.
#******************************************************************************

library(tidyverse)
library(lubridate)
library(HICtools)
library(HICwebservices)


# i. Initialising
  rm(list=ls())
  # install HICtools package from repo
  unloadNamespace("HICtools")
  install.packages("https://wl-subversion.vlaanderen.be/svn/repoSpHIC/Measurements/Packages/HICtools/HICtools_0.1.0.zip",
                   repos = NULL, type = "source")
  library("HICtools")

  unloadNamespace("HICwebservices")
  install.packages("https://wl-subversion.vlaanderen.be/svn/repoSpHIC/Webservices/R/trunk/HICwebservices_1.2.tar.gz",
                   repos = NULL, type = "source")
  library("HICwebservices")


## 1. MANUAL SETTINGS ----
  # Analysis

  minYear <- 2024; maxYear <- 2025;       # period of analysis. For one year, set minYear and maxYear to year of interest
  Quant <- c(.00,.01,.99,1)               # quantiles of interest (note: 0 = min, 1 = max); For moneos: c(.00,.01,.99,1)

  RefName <- "Antwerpen tij/Zeeschelde"   # name of reference station for tide lag, use "none" if reference is not used
  #  "Vlissingen tij/Westerschelde"|"Antwerpen tij/Zeeschelde"|"Antwerpen-Loodsgebouw"|"Vlissingen"|"zes21a-1066"

  # Thresholds for quality control
  Tr_Duration  <- 11*60                    # Time threshold for durations of rising and falling (minutes)
  Tr_Count_M   <- 706*90/100               # minimum number of observations before statistic is shown (value is agreed on 706*90/100 for 10j-overzicht)
  Tr_Count_SN  <- 21                       # minimum number of observations before statistic is shown (value is agreed on 21 for 10j-overzicht)
  Tr_Time_lag  <- 13*60                    # Time threshold for tide_get_timelag (minutes)
  Tr_Time_SN   <- 13                       # Time threshold for tide_sel_SN (hours)

  
##*************************************************************************************************************
## 2. INPUT (READ or CALL)  -----
## 2a READ data: from exports----

#DME - remove option

## 2b CALL data: Use Webservices ----
  # Er zijn in Wiski twee afzonderlijke groepen gemaakt met de hoogwaterstanden
  # en de laagwaterstanden. Het gaat om DEX_VAL_W_HW en DEX_VAL_W_LW met
  # respectievelijke group ID's 420471 en 420473.

  # Get token
  get_token()
  
  source("https://wl-subversion.vlaanderen.be/svn/repoSpHIC/Measurements/Packages/useful_functions/kiwis_call.R")
  
  group_list_ctW <- get_group_list(ts_group_id = 156202)
  selTurb        <- group_list_ctW[17,]
  
  group_list_ctW <- get_group_list(ts_group_id = 156163)
  sel <-  group_list_ctW[46,]
  
  
  ts_raw_ctW_Turb   <- get_ts_values(ts_id=53105010, from = "2024-11-18", to = "2024-11-30")
  ts_raw_ctW_WL     <- get_ts_values(ts_id=53989010, from = "2024-11-18", to = "2024-11-30")
  ts_raw_ctW_WL1min <- get_ts_values(ts_id=54018010, from = "2024-11-18", to = "2024-11-30")
  
  setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
  save(ts_raw_ctW_Turb, file="HIC_Turb_Oosterweel.Rdata")
  save(ts_raw_ctW_WL, file="HIC_WL_Antwerpen.Rdata")
  save(ts_raw_ctW_WL1min, file="HIC_WL1min_Antwerpen.Rdata")
