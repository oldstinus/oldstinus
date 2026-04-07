rm(list=ls())

#####
#####
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
load('AQP_wave_lowPT_1mb.Rdata')
load('AQP_wave_lowPTb.Rdata')
load('AQP_wave_midPT.Rdata')
load('AQP_wave_midPT_1m.Rdata')

load("Turb_1min.Rdata")
load('Turb.Rdata')

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
load('AQP_TidePTb.Rdata')

load('Vel_Vec_1s.Rdata')
load('Vec_1m.Rdata')

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ctu')
load('Vec_8Hz.Rdata')
load('Vec_8Hzc.Rdata')

#####
##### EVENT 1
time_start   <- as.POSIXct("2024-11-19 8:28:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-19 8:44:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

time_start_wav   <- as.POSIXct("2024-11-19 8:35:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop_wav    <- as.POSIXct("2024-11-19 8:40:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

tide_ind     <- 2


time_ev   <- c(time_start, time_stop)
time_wave <- c(time_start_Wav, time_stop_wav)

data_AQP1 <- AQP_wave_lowPT
data_AQP2 <- AQP_wave_midPT
data_AQP1_1m <- AQP_wave_lowPT_1m
data_AQP2_1m <- AQP_wave_midPT_1m
data_AQPt    <- AQP_tidePT

data_Vec    <- Vec_8H
data_Vecc   <- Vec_8Hc  
data_Vec1m  <- VecV_1m   
data_Vec1s  <- Vec_1s   
data_Turb   <- And_turbT

source('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/01_script_plots/Plot_events.R')
plot_wave_event("1", time_ev, time_wave, tide_ind, data_AQP1, data_AQP2, data_AQPt, data_AQP1_1m, data_AQP2_1m, data_Vec, data_Vecc, 
                data_Vec1s, data_Vec1m, data_Turb, 
                yplotlim_vel = c(-0.2,0.7), yplotlim_comp = = c(-0.6,0.2),  yplotlim_P = c(1.5,2.2))


time_start   <- as.POSIXct("2024-11-19 13:55:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-19 14:20:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

time_start_wav   <- as.POSIXct("2024-11-19 14:00:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop_wav    <- as.POSIXct("2024-11-19 14:15:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

tide_ind     <- 3
time_ev   <- c(time_start, time_stop)
time_wave <- c(time_start_wav, time_stop_wav)


plot_wave_event("2", time_ev, time_wave, tide_ind, data_AQP1, data_AQP2, data_AQPt, data_AQP1_1m, data_AQP2_1m, data_Vec, data_Vecc, 
                data_Vec1s, data_Vec1m, data_Turb, 
                yplotlim_vel = c(-0.2,0.7), yplotlim_comp = c(0,0.4), yplotlim_P = c(1.5,2.2))


time_start   <- as.POSIXct("2024-11-19 15:55:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-19 16:20:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

time_start_wav   <- as.POSIXct("2024-11-19 14:00:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop_wav    <- as.POSIXct("2024-11-19 14:15:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

tide_ind     <- 3
time_ev   <- c(time_start, time_stop)
time_wave <- c(time_start_wav, time_stop_wav)


plot_wave_event("3", time_ev, time_wave, tide_ind, data_AQP1, data_AQP2, data_AQPt, data_AQP1_1m, data_AQP2_1m, data_Vec, data_Vecc, 
                data_Vec1s, data_Vec1m, data_Turb, 
                yplotlim_vel = c(-0.2,0.7), yplotlim_comp = c(0,0.7), yplotlim_P = c(3.5,5))


time_start   <- as.POSIXct("2024-11-19 20:20:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-19 20:40:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

time_start_wav   <- as.POSIXct("2024-11-19 14:00:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop_wav    <- as.POSIXct("2024-11-19 14:15:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

tide_ind     <- 3
time_ev   <- c(time_start, time_stop)
time_wave <- c(time_start_wav, time_stop_wav)


plot_wave_event("4", time_ev, time_wave, tide_ind, data_AQP1, data_AQP2, data_AQPt, data_AQP1_1m, data_AQP2_1m, data_Vec, data_Vecc, 
                data_Vec1s, data_Vec1m, data_Turb, 
                yplotlim_vel = c(-0.2,0.7), yplotlim_comp = c(-0.2,0.4), yplotlim_P = c(1.5,2.5))


time_start   <- as.POSIXct("2024-11-20 05:00:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-20 05:20:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

time_start_wav   <- as.POSIXct("2024-11-19 14:00:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop_wav    <- as.POSIXct("2024-11-19 14:15:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

tide_ind     <- 4
time_ev   <- c(time_start, time_stop)
time_wave <- c(time_start_wav, time_stop_wav)


plot_wave_event("5", time_ev, time_wave, tide_ind, data_AQP1, data_AQP2, data_AQPt, data_AQP1_1m, data_AQP2_1m, data_Vec, data_Vecc, 
                data_Vec1s, data_Vec1m, data_Turb, 
                yplotlim_vel = c(-0.2,0.7), yplotlim_comp = c(-0.0,0.7), yplotlim_P = c(4,5))


time_start   <- as.POSIXct("2024-11-20 08:10:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-20 08:30:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

time_start_wav   <- as.POSIXct("2024-11-19 14:00:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop_wav    <- as.POSIXct("2024-11-19 14:15:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

tide_ind     <- 4
time_ev   <- c(time_start, time_stop)
time_wave <- c(time_start_wav, time_stop_wav)


plot_wave_event("6", time_ev, time_wave, tide_ind, data_AQP1, data_AQP2, data_AQPt, data_AQP1_1m, data_AQP2_1m, data_Vec, data_Vecc, 
                data_Vec1s, data_Vec1m, data_Turb, 
                yplotlim_vel = c(-0.2,0.7), yplotlim_comp = c(-0.2,0.4), yplotlim_P = c(1.5,3.5))


time_start   <- as.POSIXct("2024-11-21 05:30:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-21 05:50:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

time_start_wav   <- as.POSIXct("2024-11-19 14:00:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop_wav    <- as.POSIXct("2024-11-19 14:15:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

tide_ind     <- 6
time_ev   <- c(time_start, time_stop)
time_wave <- c(time_start_wav, time_stop_wav)


plot_wave_event("7", time_ev, time_wave, tide_ind, data_AQP1, data_AQP2, data_AQPt, data_AQP1_1m, data_AQP2_1m, data_Vec, data_Vecc, 
                data_Vec1s, data_Vec1m, data_Turb, 
                yplotlim_vel = c(-0.2,0.7), yplotlim_comp = c(-0.1,0.5), yplotlim_P = c(25,4))



inde_wav_1mL <-  which(AQP_wave_lowPT_1m[[tide_ind]]$Timestamp > time_start & AQP_wave_lowPT_1m[[tide_ind]]$Timestamp < time_stop)
inde_wav_1mM <-  which(AQP_wave_lowPT_1m[[tide_ind]]$Timestamp > time_start & AQP_wave_lowPT_1m[[tide_ind]]$Timestamp < time_stop)
inde_wav_1sL <-  which(AQP_wave_lowPT[[tide_ind]]$Timestamp > time_start & AQP_wave_lowPT[[tide_ind]]$Timestamp < time_stop)
inde_wav_1sM <-  which(AQP_wave_midPT[[tide_ind]]$Timestamp > time_start & AQP_wave_midPT[[tide_ind]]$Timestamp < time_stop)

inde_tide_5m <-  which(AQP_tidePT[[tide_ind]]$Timestamp > time_start & AQP_tidePT[[tide_ind]]$Timestamp < time_stop)

inde_vec_1m  <-  which(VecV_1m$Timestamp > time_start & VecV_1m$Timestamp < time_stop)


inde_turb <- which(And_turbT[[tide_ind]]$Timestamp > time_start & And_turbT[[tide_ind]]$Timestamp < time_stop)

###
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/3_plots/2_Events')

png(file = paste("EV1_PVEL_flow",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

#PLOT 1 - waterdepth
# AQP - waves
plot(AQP_wave_lowPT_1m[[tide_ind]]$Timestamp[inde_wav_1mL], AQP_wave_lowPT_1m[[tide_ind]]$WaterDepth[inde_wav_1mL], 
     type = 'l', ylim = c(1.5,3.5), lty = 1, col = "black")
lines(AQP_wave_midPT_1m[[tide_ind]]$Timestamp[inde_wav_1mM], AQP_wave_midPT_1m[[tide_ind]]$WaterDepth[inde_wav_1mM], 
      col = "black", lty = 2)
# AQP - tide
lines(AQP_tidePT[[tide_ind]]$Timestamp[inde_tide_5m], AQP_tidePT[[tide_ind]]$WaterDepth[inde_tide_5m], col = "orange")
# Vector
lines(VecV_1m$Timestamp[inde_vec_1m], VecV_1m$P[inde_vec_1m] - 1 , col = "blue")

#$Timestamp[ind], Velocity$Vel[ind], type = 'l',
#ylab = "Velocity [m/s]", xlab = "Time")
#abline(h = 0, col = "gray")

mtext(side = 2, text = "Waterhoogte [m]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 


#PLOT 2 - velocities
# AQP - waves
plot(AQP_wave_lowPT_1m[[tide_ind]]$Timestamp[inde_wav_1mL], AQP_wave_lowPT_1m[[tide_ind]]$Vel[inde_wav_1mL], 
     type = 'l', ylim = c(0,1.5), lty = 1, col = "black")
lines(AQP_wave_midPT_1m[[tide_ind]]$Timestamp[inde_wav_1mM], AQP_wave_midPT_1m[[tide_ind]]$Vel[inde_wav_1mM], 
      col = "black", lty = 2)
# AQP - tide
lines(AQP_tidePT[[tide_ind]]$Timestamp[inde_tide_5m], AQP_tidePT[[tide_ind]]$Vel[inde_tide_5m], col = "orange")
# Vector
lines(VecV_1m$Timestamp[inde_vec_1m], VecV_1m$Vel[inde_vec_1m], col = "blue")

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()


png(file = paste("EV1_PVEL_fhigh",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

#PLOT 1 - waterdepth
# AQP - waves
plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], AQP_wave_lowPT[[tide_ind]]$WaterDepth[inde_wav_1sL], 
     type = 'l', ylim = c(1.5,2.2), lty = 1, col = "black")
lines(AQP_wave_midPT[[tide_ind]]$Timestamp[inde_wav_1sM], AQP_wave_midPT[[tide_ind]]$WaterDepth[inde_wav_1sM], 
      col = "black", lty = 2)
# AQP - tide
lines(AQP_tidePT[[tide_ind]]$Timestamp[inde_tide_5m], AQP_tidePT[[tide_ind]]$WaterDepth[inde_tide_5m], col = "orange")
# Vector
lines(VecV_1m$Timestamp[inde_vec_1m], VecV_1m$P[inde_vec_1m] - 1 , col = "blue")

#$Timestamp[ind], Velocity$Vel[ind], type = 'l',
#ylab = "Velocity [m/s]", xlab = "Time")
#abline(h = 0, col = "gray")

mtext(side = 2, text = "Waterhoogte [m]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 


#PLOT 2 - velocities
# AQP - waves
plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], AQP_wave_lowPT[[tide_ind]]$Vel[inde_wav_1sL], 
     type = 'l', ylim = c(0,1.5), lty = 1, col = "black")
lines(AQP_wave_midPT[[tide_ind]]$Timestamp[inde_wav_1sM], AQP_wave_midPT[[tide_ind]]$Vel[inde_wav_1sM], 
      col = "black", lty = 2)
# AQP - tide
lines(AQP_tidePT[[tide_ind]]$Timestamp[inde_tide_5m], AQP_tidePT[[tide_ind]]$Vel[inde_tide_5m], col = "orange")
# Vector
lines(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$Vel[inde_wav_1sL], col = "blue")

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()

##########
## TEST ## vector
##########
ind8hz <- which(Vec_8H$Timestamp > time_start & Vec_8H$Timestamp < time_stop)

val1 <- mean(head(VecV_1m$Vel[inde_vec_1m], 2))
val2 <- mean(tail(VecV_1m$Vel[inde_vec_1m], 2))

#PLOT 2 - velocities
png(file = paste("EV1_PVEL_fTEST_VEC2b",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$Vel[inde_wav_1sL], col = "blue", type = "l", lty = 3,
     ylim = c(-0.3,0.6))
lines(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], vec_filB)

# plot(Vec_8H$Timestamp[ind8hz], Vec_8H$Vel[ind8hz] - val1, col = "blue", type = 'l',
#      ylim = c(-0.3,0.2))
plot(Vec_8H$Timestamp[ind8hz], Vec_8H$Vel[ind8hz], col = "blue", type = 'l',
     ylim = c(-0.3,0.6))
lines(Vec_8H$Timestamp[ind8hz], vec_filB8)

abline(h= val1, col = "orange")
#abline(h= val2, col = "orange")

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()


# rms
sd(vec_filB8[1:(8*60*2)])
mean(vec_filB8[1:(8*60*2)])

sd(vec_filB8[(8*60*7):(8*60*9)])



##########
## TEST ## vector
##########
ind8hz     <- which(Vec_8H$Timestamp > time_start & Vec_8H$Timestamp < time_stop)
ind8hz_sel <- which(Vec_8H$Timestamp > time_start & Vec_8H$Timestamp < time_stop)

val1 <- mean(head(VecV_1m$Vel[inde_vec_1m], 2))
val2 <- mean(tail(VecV_1m$Vel[inde_vec_1m], 2))

require(modelbased)
find_cross <- zero_crossings(vec_filB8)
test       <- find_inversions(vec_filB8)


require(oceanwaves)
wav_stat <- waveStatsZC(vec_filB8, 8, threshold = 0.02,plot = TRUE)



#PLOT 2 - velocities
png(file = paste("EV1_PVEL_fTEST_crossings",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(1,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

plot(Vec_8H$Timestamp[ind8hz], Vec_8H$Vel[ind8hz], col = "blue", type = 'l',
     ylim = c(-0.3,0.6))
lines(Vec_8H$Timestamp[ind8hz], vec_filB8)

# abline(v = Vec_8H$Timestamp[ind8hz[round(find_cross)]], col = "gray")
# abline(v = Vec_8H$Timestamp[ind8hz[round(test)]], col = "black")
abline(h = 0, col = "gray")

abline(h= wav_stat[[2]]/2, col = "orange", lty = 3)
abline(h= -wav_stat[[2]]/2, col = "orange", lty = 3)

abline(h= wav_stat[[4]]/2, col = "orange", lty = 3)
abline(h= -wav_stat[[4]]/2, col = "orange", lty = 3)

# #abline(h= val2, col = "orange")

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()





#PLOT 2 - velocities
png(file = paste("EV1_PVEL_fTEST_uvcomp",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

plot(Vec_8Hc$Timestamp[ind8hz], Vec_8Hc$U[ind8hz], col = "blue", type = 'l',
     ylim = c(-0.6,0.2))
lines(Vec_8Hc$Timestamp[ind8hz], Vec_8Hc$V[ind8hz], col = "red")

# abline(v = Vec_8H$Timestamp[ind8hz[round(find_cross)]], col = "gray")
# abline(v = Vec_8H$Timestamp[ind8hz[round(test)]], col = "black")
# abline(h = 0, col = "gray")
# 
# abline(h= wav_stat[[2]]/2, col = "orange", lty = 3)
# abline(h= -wav_stat[[2]]/2, col = "orange", lty = 3)
# 
# abline(h= wav_stat[[4]]/2, col = "orange", lty = 3)
# abline(h= -wav_stat[[4]]/2, col = "orange", lty = 3)
# 
# # #abline(h= val2, col = "orange")

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 

plot(Vec_8Hc$Timestamp[ind8hz], atan2(Vec_8Hc$V[ind8hz],Vec_8Hc$U[ind8hz]), col = "blue", type = 'l',
     ylim = c(-3,3))


mtext(side = 2, text = "Richting", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()


#PLOT 2 - velocities
png(file = paste("EV1_PVEL_fTEST_uvcomp_turb",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

plot(Vec_8Hc$Timestamp[ind8hz], Vec_8Hc$U[ind8hz], col = "blue", type = 'l',
     ylim = c(-0.6,0.2))
lines(Vec_8Hc$Timestamp[ind8hz], Vec_8Hc$V[ind8hz], col = "red")

# abline(v = Vec_8H$Timestamp[ind8hz[round(find_cross)]], col = "gray")
# abline(v = Vec_8H$Timestamp[ind8hz[round(test)]], col = "black")
# abline(h = 0, col = "gray")
# 
# abline(h= wav_stat[[2]]/2, col = "orange", lty = 3)
# abline(h= -wav_stat[[2]]/2, col = "orange", lty = 3)
# 
# abline(h= wav_stat[[4]]/2, col = "orange", lty = 3)
# abline(h= -wav_stat[[4]]/2, col = "orange", lty = 3)
# 
# # #abline(h= val2, col = "orange")

#mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 

plot(And_turbT[[tide_ind]]$Timestamp[inde_turb], And_turbT[[tide_ind]]$Turb[inde_turb], col = "blue", type = 'l',
     ylim = c(0,500))


mtext(side = 2, text = "Turbiditeit", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()

##########
## TEST ##
##########
# AQP - waves
val1 <- mean(head(VecV_1m$Vel[inde_vec_1m], 2))
val2 <- mean(tail(VecV_1m$Vel[inde_vec_1m], 2))

require(dplR)
vec_unfil   <- Vec_1s[[tide_ind]]$Vel[inde_wav_1sL] - mean(val1, val2)
vec_unfil8  <- Vec_8H$Vel[ind8hz] - mean(val1, val2)

## LOW PASS filter (for primary wave system)
vec_fil2   <- pass.filt(vec_unfil, W = c(0.1), type = "low",  method="Butterworth")

vec_filB    <- pass.filt(vec_unfil, W = c(0.1,0.5), type = "pass", method="Butterworth")
vec_filB8    <- pass.filt(vec_unfil8, W = c(0.1,0.5)/8, type = "pass", method="Butterworth")


vec_fil3   <- pass.filt(vec_unfil, W = c(0.33), type = "high",  method="Butterworth")


https://stackoverflow.com/questions/56314907/correctly-interpret-butterworth-filter-frequencies

#PLOT 2 - velocities
png(file = paste("EV1_PVEL_fTEST_0",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$P[inde_wav_1sL], col = "blue", type = "l")


plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$Vel[inde_wav_1sL], col = "blue", type = "l")
plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$Vel[inde_wav_1sL] - val1, col = "blue", type = "l", lty = 3,
     ylim = c(-0.3,0.2))
#lines(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], vec_fil, col = "black")
#lines(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], vec_fil2, col = "orange")
plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], vec_unfil - vec_fil2, col = "red", type = 'l',
     ylim = c(-0.3,0.2))

abline(h= val1, col = "orange")
abline(h= val2, col = "orange")

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()


#PLOT 2 - velocities
png(file = paste("EV1_PVEL_fTEST_0_UV",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(1,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$P[inde_wav_1sL], col = "blue", type = "l")


plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$Vel[inde_wav_1sL], col = "blue", type = "l")
plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], Vec_1s[[tide_ind]]$Vel[inde_wav_1sL] - val1, col = "blue", type = "l", lty = 3,
     ylim = c(-0.3,0.2))
#lines(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], vec_fil, col = "black")
#lines(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], vec_fil2, col = "orange")
plot(AQP_wave_lowPT[[tide_ind]]$Timestamp[inde_wav_1sL], vec_unfil - vec_fil2, col = "red", type = 'l',
     ylim = c(-0.3,0.2))

abline(h= val1, col = "orange")
abline(h= val2, col = "orange")

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()









#####
##### EVENT 2
time_start   <- as.POSIXct("2024-11-18 15:58:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-18 16:15:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

ind_event_1m <-  which(dat_AQP_wave_lowPT_1m[[1]]$Timestamp > time_start & dat_AQP_wave_lowPT_1m[[1]]$Timestamp < time_stop)
ind_event_1s <-  which(dat_AQP_wave_lowPT[[1]]$Timestamp > time_start & dat_AQP_wave_lowPT[[1]]$Timestamp < time_stop)
ind_event_5m <-  which(AQP_tidePT[[1]]$Timestamp > time_start & AQP_tidePT[[1]]$Timestamp < time_stop)


###
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/3_plots')

png(file = paste("EVENT2_PRESS_VEL",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);


#PLOT 1 - waterdepth
# AQP - waves
plot(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$WaterDepth[ind_event_1s], type = 'l')
lines(dat_AQP_wave_lowPT_1m[[1]]$Timestamp[ind_event_1m], dat_AQP_wave_lowPT_1m[[1]]$WaterDepth[ind_event_1m], lty = 2, col = "black")
# AQP - tide
lines(AQP_tidePT[[1]]$Timestamp[ind_event_5m], AQP_tidePT[[1]]$WaterDepth[ind_event_5m], col = "orange")
# Vector
lines(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], Vec_1s[[1]]$P[ind_event_1s] - 1 , col = "blue")

#$Timestamp[ind], Velocity$Vel[ind], type = 'l',
#ylab = "Velocity [m/s]", xlab = "Time")
#abline(h = 0, col = "gray")

mtext(side = 2, text = "Waterhoogte [m]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 


#PLOT 2 - velocities
# AQP - waves
plot(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$Vel[ind_event_1s], type = 'l', ylim = c(0,1))
lines(dat_AQP_wave_lowPT_1m[[1]]$Timestamp[ind_event_1m], dat_AQP_wave_lowPT_1m[[1]]$Vel[ind_event_1m], type = 'l', lty  = 2)

# AQP - tide
lines(AQP_tidePT[[1]]$Timestamp[ind_event_5m], AQP_tidePT[[1]]$Vel[ind_event_5m], col = "orange", lty = 1)

# Vector
lines(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], Vec_1s[[1]]$Vel[ind_event_1s], col = "blue")


# lines(dat_AQP_wave_lowPT_1m[[1]]$Timestamp[ind_event_1m], dat_AQP_wave_lowPT_1m[[1]]$VelCel[ind_event_1m,1], 
#       type = 'l', col = "orange", lty = 2)
# lines(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$VelCel[ind_event_1s,1], 
#       type = 'l', col = "green", lty = 2)


mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()



#####
##### EVENT 2
time_start   <- as.POSIXct("2024-11-18 17:20:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-18 17:40:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

ind_event_1m <-  which(dat_AQP_wave_lowPT_1m[[1]]$Timestamp > time_start & dat_AQP_wave_lowPT_1m[[1]]$Timestamp < time_stop)
ind_event_1s <-  which(dat_AQP_wave_lowPT[[1]]$Timestamp > time_start & dat_AQP_wave_lowPT[[1]]$Timestamp < time_stop)


###
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/3_plots')

png(file = paste("EVENT3_PRESS_VEL",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

#PLOT 1 - waterdepth
plot(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$WaterDepth[ind_event_1s], type = 'l')
lines(dat_AQP_wave_lowPT_1m[[1]]$Timestamp[ind_event_1m], dat_AQP_wave_lowPT_1m[[1]]$WaterDepth[ind_event_1m], col = "blue")

#$Timestamp[ind], Velocity$Vel[ind], type = 'l',
#ylab = "Velocity [m/s]", xlab = "Time")
abline(h = 0, col = "gray")

mtext(side = 2, text = "Waterhoogte [m]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 


#PLOT 2 - velocities
plot(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$Vel[ind_event_1s], type = 'l', ylim = c(0,1))
lines(dat_AQP_wave_lowPT_1m[[1]]$Timestamp[ind_event_1m], dat_AQP_wave_lowPT_1m[[1]]$Vel[ind_event_1m], type = 'l')

lines(dat_AQP_wave_lowPT_1m[[1]]$Timestamp[ind_event_1m], dat_AQP_wave_lowPT_1m[[1]]$VelCel[ind_event_1m,2], 
      type = 'l', col = "orange", lty = 2)
lines(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$VelCel[ind_event_1s,2], 
      type = 'l', col = "green", lty = 2)

mtext(side = 2, text = "Snelheid [m/s]", line = 1.5, cex = 0.85, font=2, outer = FALSE); 
mtext(side = 1, text = "Tijd", line = 1.5, cex = 0.85, font=2, outer = TRUE);

dev.off()







plot(dat_AQP_wave_lowPT[[1]][[7]][1:1000], dat_AQP_wave_lowPT[[1]][[8]][1:1000])





setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
load('Vel_vec.Rdata')

#####
#####
ind  <- seq(1,600,1)
ind2 <- seq(1,1200,1)


par(mfrow=c(2,1),cex.axis=0.85,mar=c(5,5,2,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

plot(Velocity$Timestamp[ind], Velocity$Vel[ind], type = 'l',
     ylab = "Velocity [m/s]", xlab = "Time")
abline(h = 0, col = "gray")
lines(Velocity_1c$Timestamp[ind2], Velocity_1c$Vel[ind2], col ='blue')
lines(Velocity_1c$Timestamp[ind2], Velocity_1c$Vel[ind2], col ='blue')µ
legend("topright", c('Dieptegemiddeld', 'Bodem'), col = c("black","blue"),
       lty= c(1,1))

plot(Velocity$Timestamp[ind], Velocity$WaterDepth[ind], type = 'l',
     ylab = "Waterdepth [m]", xlab = "Time")



library(plotly)

x <- c(1:100)
random_y <- rnorm(100, mean = 0)
data <- data.frame(x, random_y)

velplot <- data.frame(Time = seq(1,15248,1), Vel = Velocity$Vel)
velplot <- data.frame(Time = Velocity$Timestamp[1:15248], Vel = Velocity$Vel, 
                      VelCel1 = Velocity$VelCel[,1])
velplottide <- data.frame(Time = Tide$Timestamp, Vel = Tide$Vel,
                          VelC1 = Tide$VelCel[,1])

ind     <- seq(601,1200,1)

fig <- plot_ly(velplot, x = ~Time[ind], y = ~Vel[ind], 
               type = 'scatter', mode = 'lines', name = "DA")
fig <- fig %>% add_trace(y = ~VelCel1[ind] , 
                         name = "Cel1", connectgaps = TRUE)
fig <- fig %>% add_trace(data = velplottide, x = ~Time, y = ~Vel , 
                         name = "DA_total", connectgaps = TRUE)
fig <- fig %>% add_trace(data = velplottide, x = ~Time, y = ~VelC1 , 
                         name = "DA_total_C1", connectgaps = TRUE)

fig

# Save the plot to an HTML file
fig.write_html("scatter_plot.html")

orca(fig, 'plot.png')

export(fig, file = "image.png")
htmlwidgets::saveWidget(fig, file = "image.html")
plotly_IMAGE(fig, width = 500, height = 500, format = "png", scale = 2,
             out_file = "~P/test.png")