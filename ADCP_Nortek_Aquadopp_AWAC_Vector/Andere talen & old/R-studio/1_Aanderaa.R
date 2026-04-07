rm(list=ls())

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/Metingen/2024-12_09/Aanderaa_seaguardx_x1/RCM_320')

source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/read_Aanderaa.R')
require(lubridate)

###############
## Read data ##
###############
filename <- "P24_077 20241118_20241128 meting aan Galgenweel.txt"
data <- read_Aanderaa(filename)
data$Time <- data$Time - 3600

## SPLIT PER TIDE ##
msmt_period <- range(data$Time)

##
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
load('tide_Apen.Rdata')
LW  <- out[seq(1,43,2),]
len <- dim(LW)[1]-1

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
load('HIC_WL1min_Antwerpen.Rdata')
WL_1min <- ts_raw_ctW_WL1min
rm(ts_raw_ctW_WL1min)

plot(WL_1min$Timestamp[5000:10000], WL_1min$Value[5000:10000], type = 'l')
points(out$Timestamp, out$Value)


######################################
#################### change data frame
names(data)[1] <- "Timestamp"

WaterDepth_tot <- 0.10199773339984*data$Pres
WaterDepth <- 0.10199773339984*(data$Pres - 101.3)

data$WaterDepth_tot <- WaterDepth_tot
data$WaterDepth     <- WaterDepth


######################################
#################### validation on water level (time)
TR_val <- 0.75

ind_dry  <- which(WL_1min$Value < TR_val)
test     <- round_date(data$Time, "minute")
ind_sel2 <- NULL
for (i in 1:length(ind_dry)){
  
  ind_sel  <- match(test, WL_1min$Timestamp[ind_dry[i]])
  ind_sel2 <- c(ind_sel2, ind_sel)
  
}

indsel  <- test %in% WL_1min$Timestamp[ind_dry]
indsel2 <- which(indsel)

data$Turb[indsel2] <- NA
data$Temp[indsel2] <- NA
data$Pres[indsel2] <- NA


####################
## Split per tide ##
####################
out_turb <- list()
for (i in 1:len){
  
  #
  ind <- which(data$Time > LW$Timestamp[i] & data$Time < LW$Timestamp[i+1])
  out_turb[[i]] <- data[ind,]
  
}

####################
## Avg per 5 min  ##
####################
MA_1min  <- filter(data$Turb, filter = rep(1/30, 30), method = 'convolution', sides = 2)
MA_5min  <- filter(data$Turb, filter = rep(1/150, 150), method = 'convolution', sides = 2)
time_seq <- seq.int() 

ind       <- which(diff(minute(data$Time)) > 0) + 1
data_1min <- data.frame(Timestamp = data$Time[ind], Turb = MA_1min[ind])

out_turb_1min <- list()
for (i in 1:len){
  
  #
  ind <- which(data_1min$Time > LW$Timestamp[i] & data_1min$Time < LW$Timestamp[i+1])
  out_turb_1min[[i]] <- data_1min[ind,]
  
}

#############
## output  ##
#############
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
And_turb <- data
save(And_turb, file = "dat_And_turb.Rdata")

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
And_turbT    <- out_turb
And_turbT_1m <- out_turb_1min

save(And_turbT, file = "Turb.Rdata")
save(And_turbT_1m, file = "Turb_1min.Rdata")
