#check values of Jonas
rm(list=ls())

###################### read Aquadopp tidal velocity & tidal information
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
dat_AQP_tide <- load('AQP_Tide.Rdata')

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
dat_tide     <- load('tide_Apen.Rdata')

###################### find HWLW on velocity data
require(TideStatistics)
indHWLW <- HWLW_findHWLW(Velocity$WaterDepth, FLrem = TRUE, minT =  108)
dat_AQP_tide_HW <- cbind(Velocity$Timestamp[indHWLW[[1]]], Velocity$WaterDepth[indHWLW[[1]]])

  
###################### read Aquadopp tidal velocity & tidal information
plot(out$Timestamp, as.numeric(out$Value))
lines(Velocity$Timestamp, Velocity$WaterDepth, col = "blue")
points(Velocity$Timestamp[indHWLW[[1]]], Velocity$WaterDepth[indHWLW[[1]]], pch = 16, col = "blue")
points(out$Timestamp[seq(2, dim(out)[1], 2)], as.numeric(out$Value[seq(2,dim(out)[1], 2)]), pch = 17, col = "red")


######################
###################### RESULT
length(as.numeric(out$Value[seq(2,dim(out)[1], 2)]))

diffval <- as.numeric(out$Value[seq(2,dim(out)[1], 2)]) - dat_AQP_tide_HW[1:21,2]
mean(diffval) 

diffvalT  <- diff(as.numeric(out$Timestamp[seq(2,dim(out)[1], 2)]) - dat_AQP_tide_HW[1:21,1])
diffvalT2 <- difftime(out$Timestamp[seq(2,dim(out)[1], 2)],  Velocity$Timestamp[indHWLW[[1]]][1:21])



################################################
################################################
rm(list=ls())
###################### TURB
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
dat_And_turb <- load('Turb.Rdata')

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
dat_tide     <- load('tide_Apen.Rdata')

###################### find HWLW on velocity data
plot(dat_And_turbT[[2]]$Timestamp, dat_And_turbT[[2]]$WaterDepth, ylim = c(-0.5,6))
points(out$Timestamp, as.numeric(out$Value))
abline(v = out$Timestamp)



################################################
################################################
rm(list=ls())
###################### TURB
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
dat_AQP_wave <- load('AQP_wave_low_1m.Rdata')
data_1m$Timestamp <- data_1m$Timestamp[1:15248]

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
dat_tide     <- load('tide_Apen.Rdata')

indHWLW <- HWLW_findHWLW(data_1m$Depth, FLrem = TRUE, minT =  540)
dat_AQP_wave_HW <- data.frame(Timestamp = data_1m$Timestamp[indHWLW[[1]]], WaterDepth = data_1m$Depth[indHWLW[[1]]])

plot(data_1m$Timestamp, data_1m$Depth, type ="l", ylim = c(-0.5,6) )
points(out$Timestamp, as.numeric(out$Value), pch = 17, col = "blue")
points(dat_AQP_wave_HW$Timestamp, dat_AQP_wave_HW$WaterDepth, col = "green", pch = 18)       


length(as.numeric(out$Value[seq(2,dim(out)[1], 2)]))
diffval <- as.numeric(out$Value[seq(2,dim(out)[1], 2)]) - dat_AQP_wave_HW$WaterDepth[1:21]

diffvalT  <- difftime(out$Timestamp[seq(2,dim(out)[1], 2)] - dat_AQP_wave_HW$Timestamp[1:21])
diffvalT  <- diff(as.numeric(out$Timestamp[seq(2,dim(out)[1], 2)]) - as.numeric(dat_AQP_wave_HW$Timestamp[1:21]))

diffvalT2 <- difftime(as.numeric(out$Timestamp[seq(2,dim(out)[1], 2)],  Velocity$Timestamp[indHWLW[[1]]][1:21])
