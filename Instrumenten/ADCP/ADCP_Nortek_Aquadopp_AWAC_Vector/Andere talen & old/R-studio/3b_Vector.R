
# 1. Schoon de workspace
rm(list = ls())

# 2. Laad benodigde packages, stop als ze ontbreken
if (!requireNamespace("oce", quietly = TRUE)) {
  stop("Package ‘oce’ ontbreekt; installeer met install.packages('oce').")
}
if (!requireNamespace("tcltk", quietly = TRUE)) {
  stop("Package ‘tcltk’ ontbreekt; installeer met install.packages('tcltk').")
}
library(oce)
library(tcltk)

# 3. Kies werkdirectory
tk_messageBox(message = "Selecteer de werkdirectory waarin je bestanden staan")
workdir <- tk_choose.dir(default = getwd(), caption = "Kies werkdirectory")
if (is.na(workdir) || workdir == "") {
  stop("Geen werkdirectory gekozen; script wordt afgebroken.")
}
setwd(workdir)
cat("Working directory:", workdir, "\n\n")
#setwd('C:/Users/claeysst/Desktop/werkfiles/Verwerken Vector')
#list.files("C:/Users/claeysst/Desktop/werkfiles/Verwerken Vector",
#           pattern = "\\.Rdata$", full.names = TRUE)
load("Vec_vel.Rdata")
load('Vec_time.Rdata')
load('Vec_velcomp.Rdata')

load('tide_Apen.Rdata')
LW  <- out[seq(1,43,2),]
len <- dim(LW)[1]-1

####################
#################### TIME
# to UTC & remove last values which are dry
timestamp <- timestamp - 3600
ind_rem   <- which(timestamp > out$Timestamp[43])

timestamp <- timestamp[-ind_rem]

# jump in measurements
indfirst  <- which(diff(timestamp) > 1)[1]-1
timestamp <- timestamp[1:indfirst]

#######################
len       <- length(timestamp)
maxlen    <- len*8

vec_extra <- vec_s[maxlen:length(vec_s$P),]
vec_s     <- vec_s[1:maxlen,]
vec       <- vec[1:maxlen,]

# plot(vec_s$P[1:250000], type = "l")
# plot(vec_s$Vel[1:15000], type = "l")

########################
# subsample for visual inspection
indtest  <- seq(1,length(vec_s$P),8)
red_vecs <- vec_s[indtest,] 
red_vec  <- vec[indtest,] 


plot(timestamp[400000:575700], red_vecs$P[400000:575700])
ind_ev <- seq(2,length(out$Qual), 2)
abline(v = out$Timestamp[ind_ev], col = "blue", lty = 2)
ind_unev <- seq(1,length(out$Qual), 2)
abline(v = out$Timestamp[ind_unev], col = "green", lty = 1)


####################
#################### 1 sec output & full time output
t_start <- timestamp[1]
t_stop  <- timestamp[length(timestamp)]

ts_out <- seq.POSIXt(t_start, t_stop, 
                     units = "seconds", by = .125)


out_vecs <- vec_s[1:length(ts_out),]
out_vecs$Timestamp <- ts_out
Vec_8H <- out_vecs

out_vec <- vec[1:length(ts_out),]
out_vec$Timestamp <- ts_out
Vec_8Hc <- out_vec

setwd(workdir)
save(Vec_8H , file = "Vec_8Hz.Rdata")
save(Vec_8Hc , file = "Vec_8Hzc.Rdata")

##test
#format(head(ts_out), format = "%Y/%m/%d %H:%M:%OS3")

maxtide <- 0
for (i in 1:len){
  
  ind <- which(timestamp > LW$Timestamp[i])
  if (length(ind) > 0) maxtide <- maxtide + 1
}

## 3 different output frequencies (1 min, 1 sec, 8 Hz (original))
## 1 min export ##
#avg velocity => per minute => 60 measurements
#mean depth, Vel, Dir

time_out     <- ts_out[seq(1,length(ts_out),60*8)]
time_in      <- ts_out
# data_1mins_v <- avg_AQP_scal(vec_s$, time_in, time_out)
# data_1mins_p <- avg_AQP_scal(vec_s$, time_in, time_out)
MA_1min_v  <- filter(vec_s$Vel, filter = rep(1/480, 480), method = 'convolution', sides = 2)
MA_1min_p  <- filter(vec_s$Vel, filter = rep(1/480, 480), method = 'convolution', sides = 2)

#time
ind        <- which(diff(minute(time_in)) > 0) + 1
data_1min  <- data.frame(Timestamp = time_in[ind], Vel = MA_1min_v[ind], P = MA_1min_p[ind])

VecV_1m <- data_1min
setwd('C:/Users/claeysst/Desktop/werkfiles/Verwerken Vector')
save(VecV_1m , file = "Vec_1m.Rdata")


data_1m    <- list(Timestamp = time_out, u= data_1min[[1]], v=data_1min[[2]],
                   Depth = data_1mins[[1]])

data_1m$Timestamp   <- data_1m$Timestamp[1:15248]


#1s
Vec_1s <- list()
for (i in 1:maxtide){
  
  ind_1s <- which(timestamp > LW$Timestamp[i] & timestamp < LW$Timestamp[i+1])
  Vec_1s[[i]] <- data.frame(U = red_vec$U[ind_1s], V = red_vec$V[ind_1s],
                            W = red_vec$W[ind_1s], P = red_vecs$P[ind_1s], 
                            Vel = red_vecs$Vel[ind_1s]) 
  
}


#8Hz
Vec_8Hz <- list()
for (i in 1:maxtide){
  
  ind <- which(ts_out > LW$Timestamp[i] & ts_out< LW$Timestamp[i+1])
  Vec_8hz[[i]] <- data.frame(U = vec$U[ind], V = vec$V[ind],
                             W = vec$W[ind], P = vec_s$P[ind], 
                             Vel = vec_s$Vel[ind]) 
  
}


#############
## output  ##
#############
setwd(workdir)
save(Vec_1s, file = "Vel_Vec_1s.Rdata")
save(Vec_8Hz, file = "Vel_Vec.Rdata")
