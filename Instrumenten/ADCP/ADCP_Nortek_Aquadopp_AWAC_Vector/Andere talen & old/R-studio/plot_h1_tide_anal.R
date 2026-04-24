rm(list=ls())

#############
#############
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
load('tide_Apen.Rdata')
out$Value <- as.numeric(out$Value)

#############
#############
head(out)
tail(out)

mean(out$Value[which(out$Value < 2)])
mean(out$Value[which(out$Value > 2)])
range(out$Value[which(out$Value > 2)])
range(out$Value[which(out$Value < 2)])

#############
#############
time_out <- range(out$Timestamp)

t_start <- "2024-11-18 00:00:00 UTC"
t1 <- as.POSIXct(t_start, origin="1970-01-01", format = "%Y-%m-%d %H:%M:%S", tz="UTC")
t_start <- "2024-11-30 00:00:00 UTC"
t2 <- as.POSIXct(t_start, origin="1970-01-01", format = "%Y-%m-%d %H:%M:%S", tz="UTC")

tseq     <- seq(from = t1, t2, by = "days")
tseq_lab <- as.character(format(tseq, "%d/%m"))

#############
#############
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/3_plots')
png(file = paste("h1_Tide_HWLW",".png",sep=""), width=16, height=12, units="cm",res=300);
par(mfrow=c(1,1),cex.axis=0.85,mar=c(5,5,2,1),lwd=1,mgp=c(3,0.5,0), cex = 0.75);

##
xl_1=as.numeric(t1); xl_2=as.numeric(t2); xi=3600*24; 
yl_1=-0.5; yl_2=6.5; yi=0.5;
plot.new(); plot.window(xlim=c(xl_1,xl_2),ylim=c(yl_1,yl_2));box();

#axis
axis(1,at=c(seq(xl_1,xl_2,xi)),tck=-0.015,
     labels = tseq_lab, lwd.ticks=1);
axis(2,at=c(seq(yl_1,yl_2,yi)),
     tck=-0.015,las=1,lwd.ticks=1);

#grid
abline(h=seq(yl_1, yl_2, yi), v=seq(xl_1, xl_2, xi), lwd=0.65, lty=5, col="lightgray")

##
indLW <- seq(1,length(out$Value),2)
indHW <- seq(2,length(out$Value),2)
points(out$Timestamp[indLW], out$Value[indLW], pch = 1)
points(out$Timestamp[indHW], out$Value[indHW], pch = 2)

##
legend("right",
       c("Hoogwater","Laagwater"),
       col=c("black", "black"),
       pch = c(1,2), cex=0.95, box.lwd = 1, bty = "n");

##
mtext(side = 1, text = "Tijd", line = 2.5, cex = 0.85,font=2); 
mtext(side = 2, text = "Waterhoogte [mTAW]", line = 3, cex = 0.85,font=2);

##
dev.off()