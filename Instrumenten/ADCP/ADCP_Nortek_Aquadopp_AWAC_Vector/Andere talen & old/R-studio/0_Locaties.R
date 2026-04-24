rm(list=ls())

#############
#############
setwd('P:/24_077-Golfw_IGG/3_Uitvoering/Metingen/2024-12_09')

dat <- read.csv("Aquadops + golf.txt", sep = ';', header = FALSE)

##
LOC <- unique(dat$V2)

##
AVG <- NULL
for (i in 1:length(LOC)){
  
  #indices
  ind <- which(dat$V2 == LOC[i])
  
  #
  pos_avg_1    <- mean(dat$V3[ind])
  pos_avg_2    <- mean(dat$V4[ind])
  pos_avg_vert <- mean(dat$V5[ind]) 
  
  #
  ps  <- c(pos_avg_1, pos_avg_2, pos_avg_vert)
  AVG <- rbind(AVG, ps)  #cbind(LOC[i],ps))
  
}

##
AVG <- data.frame(Loc = LOC, X = AVG[,1], Y = AVG[,2], height = AVG[,3])


setwd('P:/24_077-Golfw_IGG/3_Uitvoering/4_Analyse/1_testmeting/2_data')
write.table(AVG, "Locaties_alles.txt", sep = ";", quote = FALSE)
ind <- c(2,4,5,6,8,10,12,14)
write.table(AVG[ind,], "Locaties_instr.txt", sep = ";", quote = FALSE)
