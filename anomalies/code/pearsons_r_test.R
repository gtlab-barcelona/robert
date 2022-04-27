library(raster)
library(rgdal)
library(sf)

path_data <- "C:/Users/roger/OneDrive/Documentos/Master/JAE_ICU/PNAS/"    #Path to PNAS folder
setwd(path_data)

year <- c(1994:2015)
month <- c(paste0(0,c(1:9)),10,11,12)


# LOAD V.CARDUI BMS DATAFRAMES AND KEEP ONLY SPRING DENSITIES

bms_cardui <- read.csv("raw_data/Painted_Lady_density_BMS.csv",sep=";")    #Load Hu et al's provided data (densities)
transect <- read.csv("raw_data/BMS_cat_code.csv",sep=";")                   #Load transect lenghts (retrieved from CatalanBMS webpage)

bms_cardui_trans <- merge(bms_cardui, transect, by.x = "transect_id", by.y="Transsecte") #Merge dataframes by transect code inner join

bms_cardui_trans$count <- (bms_cardui_trans$Density...individuals.m..10.6 * bms_cardui_trans$Long) / 10^6  #Calculate V.cardui counts from densities and transect lngths for each register

bms_summ <- bms_cardui_trans[which(bms_cardui_trans$Month %in% c(3,4,5)),c("Year","count","transect_id")]  #Keep only spring matching records (3,4,5 as Hu et al's considered)

bms_count <- aggregate(by=Year, data=bms_summ, FUN=sum)                      #Summarize observations by year [3,23]to delete 1988 obs and select next spring according to raster data (test 1994 NDVi with 1995 counts)
bms_count_log  <- data.frame(year=bms_count$Year,count=log10(bms_count$count))         # Apply logaritmic


#THE COMMENTED CODE BELOW IS JUST TO PREPARE RASTER FILES. ONLY THE PREPARED ONES ARE ATTACHED

# #LOAD NDVI GIMMS RASTERS
# 
# for (y in year){
#   for (mm in month[c(1,2,10)]){
#     eval(parse(text=paste0("ndvi_gimms_",mm,"_",y," <- raster('raw_data/ndvi/ndvi_gimms_africa/ndvi_gimms_africa_",mm,"_",y,".tif')")))
#     
#   }
# }
# 
# #STACK BY MONTH
# 
# #stack october
# obj <- paste0("ndvi_gimms_10_",year,collapse=", ")
# ndvi_gimms_10 <- eval(parse(text=paste0("stack(",obj,")")))
# 
# #stack january
# obj <- paste0("ndvi_gimms_01_",year,collapse=", ")
# ndvi_gimms_01 <- eval(parse(text=paste0("stack(",obj,")")))
# 
# #stack february
# obj <- paste0("ndvi_gimms_02_",year,collapse=", ")
# ndvi_gimms_02 <- eval(parse(text=paste0("stack(",obj,")")))
# 
# #LOAD KERNELS AND CROP AREA
# 
# kern <- readOGR("prep_data/model_mask/africa/kernels.shp")
# 
# ndvi_gimms_oct <- mask(crop(ndvi_gimms_10,extent(kern)),kern)
# names(ndvi_gimms_oct) <- c(1994:2015)
# writeRaster(ndvi_gimms_oct,"prep_data/ndvi_gimms_oct.tif")
# 
# ndvi_gimms_jan <- mask(crop(ndvi_gimms_01,extent(kern)),kern)
# names(ndvi_gimms_jan) <- c(1994:2015)
# writeRaster(ndvi_gimms_jan,"prep_data/ndvi_gimms_jan.tif")
# 
# ndvi_gimms_feb <- mask(crop(ndvi_gimms_02,extent(kern)),kern)
# names(ndvi_gimms_feb) <- c(1994:2015)
# writeRaster(ndvi_gimms_feb,"prep_data/ndvi_gimms_feb.tif")


#Load cropped by kernels and stacked raster time-series

for (i in c("oct","jan","feb")){                                         #Jan and Feb to replicate, and October tot test, I can attach other months if needed
  eval(parse(text=paste0("ndvi_gimms_",i," <- stack('raw_data/ndvi_gimms_",i,".tif')")))
  eval(parse(text=paste0("names(ndvi_gimms_",i,") <- 1994:2015")))
}


#TURN RASTERS TO DATA FRAMES

for (y in year[c(1:22)]){                      #[1,21] bc we don't have 2016's V.cardui data, so can't test 2015 NDVI
  for(m in c("jan","feb","oct")){
    
  cat("WORKING ON ==>",y,"\n")
    
  eval(parse(text=paste0("df_",y,"_",m," <- as.data.frame(ndvi_gimms_",m,"$X",y,",xy=TRUE)")))
  eval(parse(text=paste0("names(df_",y,"_",m,") <- c('x','y','ras_values')")))
  }
}


#MERGE DATA FRAMES ALL IN ONE

for (m in c("jan","feb","oct")){
  n <- paste0("df_",1994:2015,"_",m,"[3]",collapse=", ")
  eval(parse(text=paste0("raster_values_",m," <- data.frame(",n,")")))
  eval(parse(text=paste0("colnames(raster_values_",m,") <- 1994:2014")))
}



#TEST CORRELATIONS AND BUILD RASTERS AGAIN

bms_corr <- data.frame("pearson_r","p_value")
ras <- raster_values_jan                                  # <=== ADD THE MONTH TO TEST CORRELATIONS

for (n in 1:nrow(ras)){
 if (is.na(ras[n,1])){
   
   bms_corr[n,] <- NA
   
   cat(n, "SET TO NA", "\n" )
   
 }else{
   
   bms_corr[n,1] <- cor.test(bms_count$count[c(1:15,17:22)],as.numeric(ras[n,c(1:15,17:22)]),method = "pearson")$estimate   
   bms_corr[n,2] <- cor.test(bms_count$count[c(1:15,17:22)],as.numeric(ras[n,c(1:15,17:22)]),method = "pearson")$p.value
   
   cat(n,"OK","\n")
 }
}



ifelse(ras == raster_values_jan, bms_corr_jan <- bms_corr, bms_corr_log_feb <- bms_corr)


#TURN NEW DATA FRAMES TO RASTERS WITH ORIGINAL GEOMETRY

bms_corr$X.pearson_r. <- as.numeric(bms_corr$X.pearson_r.)
bms_corr$X.p_value. <- as.numeric(bms_corr$X.p_value.)

bms_corr$x <- df_1994_jan$x      #Same for all raster -> data.frame conversions
bms_corr$y <- df_1994_jan$y

spg_spring <- bms_corr
 
coordinates(spg_spring) <- ~ x + y
 
gridded(spg_spring) <- TRUE
 
pearson_test_log_bms_ndvi<- stack(spg_spring)
 
crs(pearson_test_log_bms_ndvi) <- crs(ndvi_gimms_jan) #same for all rasters

plot(pearson_test_log_bms_ndvi)
 
writeRaster(pearson_test_bms_ndvi,"results/corr_test_bms_count_spring_ndvi_jan_.tif",overwrite=TRUE)
