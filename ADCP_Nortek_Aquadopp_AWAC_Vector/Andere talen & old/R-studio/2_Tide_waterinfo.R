rm(list=ls())

###############################
############################### READ DATA
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data/Antwerpen tij_Zeeschelde_Waterpeil getij (1)')

dat <- read.csv2("Antwerpen tij_Zeeschelde_Waterpeil getij.csv", skip = 8, header = TRUE)
dat <- dat[,c(1,2,3)]

############################### CONVERT TO DATE/TIME
############################### 
#### afwaarts
## TIME
data_time <- as.character(dat$X.Timestamp)

## DATE
test  <- strsplit(data_time,'T')
test2 <- sapply(test,'[[',2)
test3 <- strsplit(test2,'[.]')
test4 <- sapply(test3,'[[',2)
test5 <- strsplit(test4,'[+]')
test6 <- sapply(test5,'[[',2)

date <- sapply(test,'[[', 1)
hour <- sapply(test3,'[[', 1)
timezone <- test6

datetime <- NULL
datetime <- as.POSIXct(paste(date,hour), format="%Y-%m-%d %H:%M:%S", tz="UTC")
datetime[which(timezone =="01:00")] <- datetime[which(timezone =="01:00")] - 3600 ;
datetime[which(timezone =="02:00")] <- datetime[which(timezone =="02:00")] - 7200 ;
datetime[which(timezone =="00:00")] <- datetime[which(timezone =="00:00")];
#if (! timezone == "01:00" | timezone == "02:00" | timezone == "00:00") stop('Strange time zone ...')

date <- as.Date(datetime)
rm(test, test3, test5, test2, test4, test6, timezone, hour, data_time)

###########
###########
out <- data.frame(Timestamp = datetime, Date = date, Value = dat$Value, Qual = dat$Quality.Code)
out <- out[2:44,]

setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
save(out, file = "tide_Apen.Rdata")