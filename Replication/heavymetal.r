load("./data/heavymetal.RData") 

HMs = purrr::map(HMImages,terra::rast) |> 
  terra::rast()
names(HMs) = c("cu","cd","mg","pb")

Envs = purrr::map(EnviImages,terra::rast) |> 
  terra::rast()
names(Envs) = c("ntl","industry")

heavymetal = c(HMs,Envs)
terra::plot(heavymetal)

# terra::writeRaster(heavymetal,'./data/heavymetal.tif', datatype = "FLT8S", overwrite = TRUE)
heavymetal = terra::rast('./data/heavymetal.tif')

g1 = spEDM::gccm(heavymetal, "ntl", "cu",
                 libsizes = matrix(rep(seq(10,120,20),time = 2),ncol = 2),
                 E = 4, k = 5, style = 0, stack = TRUE,
                 pred = as.matrix(expand.grid(seq(5,125,5), seq(5,130,5))),
                 dist.metric = "L1", dist.average = FALSE, detrend = FALSE)

g2 = spEDM::gccm(heavymetal, "industry", "cu",
                 libsizes = matrix(rep(seq(10,120,20),time = 2),ncol = 2),
                 E = 4, k = 5, style = 0, stack = TRUE,
                 pred = as.matrix(expand.grid(seq(5,125,5), seq(5,130,5))),
                 dist.metric = "L1", dist.average = FALSE, detrend = FALSE)

g3 = spEDM::gpc(heavymetal,"cu","ntl",E = 3, tau = 1, k = 4)
