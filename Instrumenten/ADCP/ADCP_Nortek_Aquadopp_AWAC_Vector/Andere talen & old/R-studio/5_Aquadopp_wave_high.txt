rm(list=ls())

require(oce)
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/Check_Amp.R')
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/Read_AQP_Tide.R')
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/Clean_AQP_Tide.R')
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/Cal_Vel_DA.R')
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/Cal_Vel_fixC.R')
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/Deg2Rad_Unit.R')
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/avg_AQP_prof.R')
source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/0_scripts/avg_AQP_scal.R')

## set WLevels
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
load('HIC_WL1min_Antwerpen.Rdata')
WL_1min <- ts_raw_ctW_WL1min
rm(ts_raw_ctW_WL1min)

## set velocity dir
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/Metingen/2024-12_09/Aquadopp_x3/11972_laag_300s_60s')
setwd('C:/Users/meiredk/Desktop/Aquadopp_x3/13794_midden_1Hz')

# find basic file
file <- list.files(".",pattern=".PRF$")

###############
## Read data ##
###############
tresholdval <- 3
valD        <- 1.5

# Read data
data           <- read_AQP_tide(file, tresholdval, valD)

data$Timestamp <- data$Timestamp - 3600
thresh_WL      <- 1.28

# Clean data
data2           <- Clean_AQP_data(data, WLdata = WL_1min, thresh_WL)
data2           <- data2[-c(7,8)]

###
###
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
save(data2, file = "Rawdata_AQPmid.Rdata")

load('Rawdata_AQPmid.Rdata')
data <- data2
rm(data2)

names(data)[5] <- "WaterDepth"

#avg velocity => per minute => 60 measurements
#mean depth, Vel, Dir
time_out   <- data$Timestamp[seq(1,length(data$Timestamp),60)]
time_in    <- data$Timestamp
data_1min  <- avg_AQP_prof(data[c(2,3)], time_in, time_out)
data_1mins <- avg_AQP_scal(data[5], time_in, time_out)
data_1m    <- list(Timestamp = time_out, u= data_1min[[1]], v=data_1min[[2]],
                   Depth = data_1mins[[1]])

data_1m$Timestamp   <- data_1m$Timestamp[1:9912]

#avg velocity => per minute => 60 measurements
#mean depth, Vel, Dir
attach(data_1m)

# Calculate average values
Velocity <- Cal_Vel_DA(u, v, Depth, data$DistCells, Cell_up = 1, Cell_down = 0, ind_AMP)

detach(data_1m)

# Velocity &
Velocity$Dir    <- Deg2Rad_Unit(Velocity$Dir, 2, 2)
Velocity$DirCel <- Deg2Rad_Unit(Velocity$DirCel, 2, 2)

# add ...
Velocity$WaterDepth  <- data_1m$Depth
Velocity$heightCells <- data$DistCells
Velocity$Timestamp   <- data_1m$Timestamp 

AQP_wave_mid_1m <- Velocity

setwd('C:/Users/meiredk/Desktop/Aquadopp_x3/anal_Laag_1Hz')
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
save(AQP_wave_mid_1m, file = "AQP_wave_mid_1m.Rdata")


#######################
#######################
# PER TIDE
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
load('tide_Apen.Rdata')
LW  <- out[seq(1,43,2),]
len <- dim(LW)[1]-1

####################
## Split per tide ##
####################
out_1m_PT  <- list()
out_1m_PTl <- list()
for (i in 1:len){
  
  #
  ind <- which(AQP_wave_mid_1m$Timestamp > LW$Timestamp[i] & AQP_wave_mid_1m$Timestamp < LW$Timestamp[i+1])
  for (j in 1:9){
    
    if (j < 5) out_1m_PT[[j]] <- AQP_wave_mid_1m[[j]][ind]
    if (j > 4 & j < 7) out_1m_PT[[j]] <- AQP_wave_mid_1m[[j]][ind,]
    if (j == 7) out_1m_PT[[j]] <- AQP_wave_mid_1m[[j]][ind]
    if (j == 8) out_1m_PT[[j]] <- AQP_wave_mid_1m[[j]]
    if (j == 9) out_1m_PT[[j]] <- AQP_wave_mid_1m[[j]][ind]
    
  }
  names(out_1m_PT) <- names(AQP_wave_mid_1m)
  out_1m_PTl[[i]] <- out_1m_PT
  
}

AQP_wave_midPT_1m <- out_1m_PTl
setwd('C:/Users/meiredk/Desktop/Aquadopp_x3/anal_Laag_1Hz')
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
save(AQP_wave_midPT_1m, file = "AQP_wave_midPT_1m.Rdata")



#########################
#########################
##### PROFILES ##########
#########################
#########################
# FULL TIME
###########
#avg velocity => per minute => 60 measurements
#mean depth, Vel, Dir
attach(data)

# Calculate average values
Velocity <- Cal_Vel_DA(u, v, WaterDepth, DistCells, Cell_up = 1, Cell_down = 0, ind_AMP)

detach(data)

# Velocity &
Velocity$Dir    <- Deg2Rad_Unit(Velocity$Dir, 2, 2)
Velocity$DirCel <- Deg2Rad_Unit(Velocity$DirCel, 2, 2)

# add ...
Velocity$WaterDepth  <- data$WaterDepth
Velocity$heightCells <- data$DistCells
Velocity$Timestamp   <- data$Timestamp

AQP_wave_mid <- Velocity

setwd('C:/Users/meiredk/Desktop/Aquadopp_x3/anal_Laag_1Hz')
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
save(AQP_wave_mid, file = 'AQP_wave_mid.Rdata')

####################
## Split per tide ##
####################
out_PT  <- list()
out_PTl <- list()
for (i in 1:len){
  
  #
  ind <- which(AQP_wave_mid$Timestamp > LW$Timestamp[i] & AQP_wave_mid$Timestamp < LW$Timestamp[i+1])
  for (j in 1:9){
    if (j > 1 & j < 5) out_PT[[j]] <- AQP_wave_mid[[j]][ind]
    if (j > 4 & j < 7) out_PT[[j]] <- AQP_wave_mid[[j]][ind,]
    if (j == 7) out_PT[[j]] <- AQP_wave_mid[[j]][ind]
    if (j == 8) out_PT[[j]] <- AQP_wave_mid[[j]]
    if (j == 9) out_PT[[j]] <- AQP_wave_mid[[j]][ind]
    
  }
  
  names(out_PT) <- names(AQP_wave_mid)
  out_PTl[[i]] <- out_PT
  
}

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
AQP_wave_midPT <- out_PTl
save(AQP_wave_midPT, file = "AQP_wave_midPT.Rdata")








###########
# 1min
###########



#########################
#########################
####### CELLLS ##########
#########################
#########################
#CELL BOTTOM
###########
# 1min
###########
#avg velocity => per minute => 60 measurements
#mean depth, Vel, Dir
attach(data_1m)

# Calculate average values
Velocity_1c <- Cal_Vel_fixC(u, v, Depth, data$DistCells, CellS =1)

detach(data_1m)

# Velocity &
Velocity_1c$Dir    <- Deg2Rad_Unit(Velocity_1c$Dir, 2, 2)

# add ...
Velocity_1c$Timestamp   <- data_1m$Timestamp
Velocity_1c$WaterDepth  <- data_1m$Depth
Velocity_1c$heightCells <- data$DistCells

setwd('C:/Users/meiredk/Desktop/Aquadopp_x3/anal_Laag_1Hz')
save(Velocity_1c, file = 'AQP_Wave_Low_C1_1min.Rdata')

#############
## output  ##
#############
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
setwd('C:/Users/meiredk/Desktop/Aquadopp_x3/anal_Laag_1Hz')
save(Velocity, file = 'AQP_Wave_Low_P_Full.Rdata')
save(Velocity1m, file = 'AQP_Wave_Low_P_1m.Rdata')

plo