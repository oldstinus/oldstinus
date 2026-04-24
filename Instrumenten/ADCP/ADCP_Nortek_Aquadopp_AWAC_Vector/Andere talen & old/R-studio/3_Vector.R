# -----------------------------------------------
# Script: Inlezen vector‐ en tijddatasets met tcltk GUI
# Doel:   gebruiker kiest werkmap en bestanden interactief
# Auteur: (jouw naam)
# Datum:  2025-05-19
# -----------------------------------------------

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

###############
## Lees tijd ##
###############
# 4. Kies .sen-bestand
tk_messageBox(message = "Selecteer het .sen-bestand (bijv. KANNE05.sen)")
sen_files <- tk_choose.files(default = workdir,
                             caption = "Selecteer .sen-bestand",
                             multi = FALSE,
                             filters = matrix(c("sen files", ".sen",
                                                "All files", "*"),
                                              ncol = 2, byrow = TRUE))
if (length(sen_files)==0 || !file.exists(sen_files)) {
  stop("Geen geldig .sen-bestand gekozen.")
}
file_sen <- sen_files[1]
cat("Inlezen .sen-bestand:", file_sen, "\n")

# 5. Bepaal automatisch het aantal kolommen en lees in
#    - fill=TRUE vult kortere rijen aan met NA
dat_tim <- read.table(file_sen,
                      header    = FALSE,
                      fill      = TRUE,
                      na.strings= c("NA",""))
n_sen    <- ncol(dat_tim)
colnames(dat_tim) <- paste0("V", seq_len(n_sen))

# 6. Pak de eerste zes kolommen als maand, dag, jaar, uur, minuut, seconde
dayval <- with(dat_tim,
               paste(V1, V2, V3, sep = "/"))
timval <- with(dat_tim,
               sprintf("%02d:%02d:%02d", V4, V5, V6))
timestamp <- as.POSIXct(paste(dayval, timval),
                        format = "%m/%d/%Y %H:%M:%S",
                        tz     = "UTC")

# 7. Sla timestamp op
save(timestamp, file = "Vec_time.Rdata")
cat("Tijdstempels opgeslagen in Vec_time.Rdata\n\n")
rm(dat_tim, dayval, timval)

###############
## Lees metingen ##
###############
# 8. Kies .dat-bestand
tk_messageBox(message = "Selecteer het .dat-bestand (bijv. KANNE05.dat)")
dat_files <- tk_choose.files(default = workdir,
                             caption = "Selecteer .dat-bestand",
                             multi = FALSE,
                             filters = matrix(c("dat files", ".dat",
                                                "All files", "*"),
                                              ncol = 2, byrow = TRUE))
if (length(dat_files)==0 || !file.exists(dat_files)) {
  stop("Geen geldig .dat-bestand gekozen.")
}
file_dat <- dat_files[1]
cat("Inlezen .dat-bestand:", file_dat, "\n")

# 9. Lees het .dat-bestand en geef generieke kolomnamen
dat_val <- read.table(file_dat, header = FALSE, fill = TRUE, na.strings = c("NA",""))
n_dat   <- ncol(dat_val)
colnames(dat_val) <- paste0("V", seq_len(n_dat))

# 10. Bouw je vector‐data.frames
vec   <- data.frame(
  U = dat_val$V3,
  V = dat_val$V4,
  W = dat_val$V5
)
vec_s <- data.frame(
  P   = dat_val$V15,
  Vel = sqrt(dat_val$V3^2 + dat_val$V4^2)
)

# 11. Sla snelheids- en drukscores op
save(vec_s, file = "Vec_vel.Rdata")
save(vec,   file = "Vec_velcomp.Rdata")
cat("Data opgeslagen in Vec_vel.Rdata en Vec_velcomp.Rdata\n")
