rm(list=ls())

#####
#####
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
load('AQP_wave_lowPT_1m.Rdata')
load('AQP_wave_lowPT.Rdata')

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/0_summary_ptide')
load('AQP_TidePT.Rdata')

load('Vel_Vec_1s.Rdata')

#####
##### EVENT 1
time_start   <- as.POSIXct("2024-11-18 13:02:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
time_stop    <- as.POSIXct("2024-11-18 13:19:00", tz = "UTC", format = "%Y-%m-%d %H:%M:%S")


ind_event_1m <-  which(dat_AQP_wave_lowPT_1m[[1]]$Timestamp > time_start & dat_AQP_wave_lowPT_1m[[1]]$Timestamp < time_stop)
ind_event_1s <-  which(dat_AQP_wave_lowPT[[1]]$Timestamp > time_start & dat_AQP_wave_lowPT[[1]]$Timestamp < time_stop)
ind_event_5m <-  which(AQP_tidePT[[1]]$Timestamp > time_start & AQP_tidePT[[1]]$Timestamp < time_stop)


###
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/3_plots')

png(file = paste("EVENT1_PRESS_VEL",".png",sep=""), width=24, height=12, units="cm",res=300);
par(mfrow=c(2,1),cex.axis=0.85, oma = c(4,4,2,0.5),
    mar=c(1,1,1,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

#PLOT 1 - waterdepth
# AQP - waves
plot(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$WaterDepth[ind_event_1s], type = 'l', ylim = c(1.7,2.5))
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
lines(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], Vec_1s[[1]]$Vel[ind_event_1s] , col = "blue")


# lines(dat_AQP_wave_lowPT_1m[[1]]$Timestamp[ind_event_1m], dat_AQP_wave_lowPT_1m[[1]]$VelCel[ind_event_1m,1], 
#       type = 'l', col = "orange", lty = 2)
# lines(dat_AQP_wave_lowPT[[1]]$Timestamp[ind_event_1s], dat_AQP_wave_lowPT[[1]]$VelCel[ind_event_1s,1], 
#       type = 'l', col = "green", lty = 2)

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