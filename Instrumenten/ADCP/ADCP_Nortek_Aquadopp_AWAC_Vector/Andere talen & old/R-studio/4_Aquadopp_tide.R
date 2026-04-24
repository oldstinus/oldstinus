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

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/Metingen/2024-12_09/Aquadopp_x3/11972_laag_300s_60s')

# find basic file
file <- list.files(".",pattern=".prf$")

###############
## Read data ##
###############
thresholdval <- 3
valD         <- 1.5

# Read data
data <- read_AQP_tide(file, tresholdval, valD)

data$Timestamp <- data$Timestamp - 3600
thresh_WL      <- 0.61

# Clean data
data2           <- Clean_AQP_data(data, WLdata = WL_1min, thresh_WL)
data2           <- data2[-c(7,8)]

###
###
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
save(data2, file = "Rawdata_Tide.Rdata")
data <- data2
rm(data2)


attach(data)

# Calculate average values
Velocity <- Cal_Vel_DA(u, v, Depth, DistCells, Cell_up = 1, Cell_down = 0, ind_AMP)

detach(data)

# Velocity &
Velocity$Dir    <- Deg2Rad_Unit(Velocity$Dir, 2, 2)
Velocity$DirCel <- Deg2Rad_Unit(Velocity$DirCel, 2, 2)

# add ...
Velocity$WaterDepth  <- data$Depth
Velocity$heightCells <- data$DistCells
Velocity$Timestamp   <- data$Timestamp
  
AQP_Tide <- Velocity

#############
#############
## output  ##
#############
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
save(AQP_Tide, file = 'AQP_Tideb.Rdata')

#######################
#######################
# PER TIDE
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
load('tide_Apen.Rdata')
LW  <- out[seq(1,43,2),]
len <- dim(LW)[1]-1

#############
#############
## output  ##
#############
out_PT  <- list()
out_PTl <- list()
for (i in 1:len){
  
  #
  ind <- which(AQP_Tide$Timestamp > LW$Timestamp[i] & AQP_Tide$Timestamp < LW$Timestamp[i+1])
  for (j in 1:9){
    
    if (j < 5) out_PT[[j]] <- AQP_Tide[[j]][ind]
    if (j > 4 & j < 7) out_PT[[j]] <- AQP_Tide[[j]][ind,]
    if (j == 7) out_PT[[j]] <- AQP_Tide[[j]][ind]
    if (j == 8) out_PT[[j]] <- AQP_Tide[[j]]
    if (j == 9) out_PT[[j]] <- AQP_Tide[[j]][ind]
    
  }
  names(out_PT) <- names(AQP_Tide)
  out_PTl[[i]] <- out_PT
  
}

AQP_tidePT <- out_PTl
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
save(AQP_tidePT, file = "AQP_TidePTb.Rdata")
