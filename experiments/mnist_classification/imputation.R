## ----setup, include = FALSE--------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(cache = TRUE, echo=TRUE, eval = TRUE)


## ----------------------------------------------------------------------------------------------------------------------------------------------
require(knitr)
#purl("v11.Rmd", output = 'v11.R')


## ----------------------------------------------------------------------------------------------------------------------------------------------
packages <- c(
  "missMDA", 
  "softImpute", 
  "caret", 
  "caTools", 
  "glue", 
  "jsonlite", 
  "future.apply", 
  "dslabs", 
  "cowplot", 
  "magick", 
  "progress"
)
# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE)) 

plan(multisession, workers = 8)  


## ---- message = FALSE--------------------------------------------------------------------------------------------------------------------------
# require(missMDA)
# require(softImpute)
# require(mice)
# require(missForest)
# require(caret)
# require(caTools) # to create train/test split 
# require(e1071)
# require(glue)
# require(jsonlite)
# require(future.apply)
# 
# require("cowplot")
# require("magick") 
# #plan(multisession)
#   
# require(dslabs)  


## ----------------------------------------------------------------------------------------------------------------------------------------------



## ----------------------------------------------------------------------------------------------------------------------------------------------


#getting the path to save 

curr_dir = getwd()
path = '../../data/mnist/raw/'

mnist_path = file.path(curr_dir, path) 
print(mnist_path)

if (!file.exists(file.path(mnist_path, "train-images-idx3-ubyte")) |
  !file.exists(file.path(mnist_path, "t10k-images-idx3-ubyte")) |
  !file.exists(file.path(mnist_path, "train-labels-idx1-ubyte")) |
  !file.exists(file.path(mnist_path, "t10k-labels-idx1-ubyte")) 
  ){
  
  # getting the data 
  mnist <- read_mnist(
    path = NULL,
    destdir = mnist_path, 
    download = TRUE,
    url = "https://www2.harvardx.harvard.edu/courses/IDS_08_v2_03/",
    keep.files = TRUE
  )  
  
  # clear folder data (avoid wrong zipping)
  list_files = list.files(path=mnist_path) 
  for (x in 1:length(list_files)){
    file_path = file.path(mnist_path, list_files[x]) 
    if (substring(file_path, nchar(file_path)-2, nchar(file_path)) == '.gz'){
      R.utils::gunzip(file_path, overwrite=TRUE, remove=FALSE) 
    }
    }  
} 
  


## ----------------------------------------------------------------------------------------------------------------------------------------------
# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
} 



## ----------------------------------------------------------------------------------------------------------------------------------------------
# load images
processing_mnist_data <- function (){
  train = load_image_file(file.path(mnist_path, "train-images-idx3-ubyte"))
  test  = load_image_file(file.path(mnist_path,"t10k-images-idx3-ubyte")) 

  train$label =  as.factor(load_label_file(file.path(mnist_path,"train-labels-idx1-ubyte")))

  test$label = as.factor(load_label_file(file.path(mnist_path,"t10k-labels-idx1-ubyte")))
  result = list('train'=train, 'test'=test)
  return(result)
}


## ----------------------------------------------------------------------------------------------------------------------------------------------
processed_data = processing_mnist_data()
train = processed_data$train 
test = processed_data$test
X.train = train[, -785]
X.test = test[, -785]
y.train = train[, 785, drop=F]
y.test = test[, 785, drop=F] 


## ----------------------------------------------------------------------------------------------------------------------------------------------
find_cov_ij <- function(Xij, Sii, Sjj){
  # Xij: the i, j column of the original matrix
  # sii, sjj = \hat{Sigma}_{ii}, \hat{Sigma}_{jj}
  # s11 = sum(Xij[,1]**2, na.rm = TRUE)
  # s12 = sum(Xij[,1]*Xij[,2], na.rm = TRUE)
  # s22 = sum(Xij[,2]**2, na.rm = TRUE)
  #start edited--------------------------
  comlt_Xij = Xij[complete.cases(Xij), ] 
  s11 = sum(comlt_Xij[,1]**2)
  s12 = sum(comlt_Xij[,1]*comlt_Xij[,2])
  s22 = sum(comlt_Xij[,2]**2)
  #end edited-------------------------- 
  m = sum(complete.cases(Xij))
  coef = c(s12*Sii*Sjj, 
                 m*Sii*Sjj-s22*Sii-s11*Sjj,
                 s12, -m)
  sol = polyroot(z = coef)
  sol = Re(sol)
  scond = Sjj - sol^2/Sii
  
  #Sii >0 
  #start edited--------------------------
  #etas = suppressWarnings(-m*log(sol) - (Sjj-2*sol/Sii*s12+sol^2/Sii^2*s11)/scond)
  etas = suppressWarnings(-m*log(scond) - (Sjj-2*sol/Sii*s12+sol^2/Sii^2*s11)/scond) 
  #end edited--------------------------
  return(sol[which.max(etas)])
}

dpers <- function(Xscaled){
  # Xscaled: scaled input with missing data
  # THE INPUT MUST BE NORMALIZED ALREADY
  shape = dim(Xscaled) # dimension
  S = matrix(0, shape[2],shape[2])
  diag(S) = apply(Xscaled, 2, function(x) var(x, na.rm=TRUE))
  # Get the index of the upper triangular matrix (row, column)
  Index<-which(upper.tri(S,diag=FALSE),arr.ind=TRUE)
  # compute the covariance and assign to S based on Index
  #start edited-------------------------- 
  total = nrow(Index)  
  pb <- txtProgressBar(min = 0, max = total, style = 3)   
  find_cov_upper_triag = function(i) {
    setTxtProgressBar(pb, i) 
    if (S[Index[i,1], Index[i,1]] == 0 | S[Index[i,2], Index[i,2]] == 0){
      return(NA)
    }
    else{
      return (
        find_cov_ij(
            Xscaled[,c(Index[i,1],Index[i,2])], 
            S[Index[i,1], Index[i,1]], 
            S[Index[i,2], Index[i,2]]
            )
      )
    }
  }
  numCores = availableCores() 
  plan(multisession, workers = numCores)  
  S_upper_calc = unlist(future_lapply(1:total, find_cov_upper_triag))
  
  stopifnot(length(S_upper_calc) == length(S[Index]))
  #end edited-------------------------- 
  S[Index]  = S_upper_calc 
  S = S + t(S)
  diag(S) = diag(S)/2
  return(S)
} 


## ----------------------------------------------------------------------------------------------------------------------------------------------
impDi <- function(S, Xtest, threshold, nlargest=2){
  print(threshold)
  Xtest[is.na(Xtest)] <- NaN   
  Xpred_original = Xtest
  
  #not to apply the algorithm on the ZERO VARIANCE columns, just fill the with 0 / or mean value (which is also 0)
  non_zero_var = (which(diag(S)!=0)) 
  
  Xtest = Xpred_original[, non_zero_var, drop=F] 
  Xpred = Xpred_original[, non_zero_var, drop=F] 
  S = S[non_zero_var, non_zero_var, drop=F]
  
  missingCols =  which(colSums(is.na(Xtest)) > 0) 
  DIMV1feature <- function(f){
    setF = which(abs(S[,f]) >= threshold)  
    #setF : Col with high corr with f
    #Fos need to be exist setF need to have at least one pair with different missing pattern f
    naRows = which(is.na(Xtest[,f]))

    allSameMissingPatternRemoving <- function(setF){
        return(sum(colSums(is.na(Xtest[naRows, setF, drop=F])) == length(naRows))==length(setF)) 
    }
    
    #remove columns have exactly same missing pattern with f  
    sameMissingPatternCheck = allSameMissingPatternRemoving(setF)
    #if there only 1 col have high correlation with f, and it has same missing pattern with f then find 1 other column have the highest correlation with f (and of course  corr lower than threshold)
    if((sameMissingPatternCheck == TRUE) | (length(setF)==1)){
      samePatternFeatures = which(colSums(is.na(Xtest[naRows, ])) == length(naRows))  
      tempS = S; diag(tempS) <- NA; tempS[, samePatternFeatures] <- NA; 
      res <- order(tempS[,f,drop=F], decreasing = TRUE)[seq_len(nlargest)]; 
      setF = arrayInd(res, dim(tempS[,f,drop=F]), useNames = TRUE)[, 1]
    }

    Df_row_pool = which(is.na(Xtest[, f])) 
    
    # after having setF and f, we then find similar missing pattern in each row 
    while (length(Df_row_pool) > 0){
      s = Df_row_pool[1]
      Fos = intersect(which(!is.na(Xtest[s,,drop=F])), setF)
      Fms = intersect(which(is.na(Xtest[s,,drop=F])), setF) 
      
      #6 lines below is to calculate Z 
      maskOfXtestFilterFos = matrix(0, dim(Xtest)[1], dim(Xtest)[2]) 
      maskOfXtestFilterFos[Df_row_pool, Fos] = Xtest[Df_row_pool, Fos] 
      Z_row_ids_fiter_observed = intersect(which(rowSums(is.na(maskOfXtestFilterFos[, Fos,drop=F]))==0), Df_row_pool)
      maskOfXtestFilterFms = matrix(0, dim(Xtest)[1], dim(Xtest)[2])
      maskOfXtestFilterFms[Z_row_ids_fiter_observed, Fms] = Xtest[Z_row_ids_fiter_observed, Fms] 
      Z_row_ids_fiter_missing = which(rowSums(is.na(maskOfXtestFilterFms))==length(Fms))  
      
      Z_row_ids = Z_row_ids_fiter_missing
      
      So = S[Fos, Fos]
      Smo = S[Fos, f] 
      beta = solve(So) %*% Smo
      
      Xtest[Z_row_ids, f] = t(beta) %*% t(Xtest[Z_row_ids, Fos])
      Df_row_pool <- setdiff(Df_row_pool, Z_row_ids)  
      
    }
    return(Xtest[, f])
  }
  # run the DIMV1feature parallely on every single features  
  Xpred_result= future_lapply(missingCols, DIMV1feature)
  Xpred[, missingCols] = t(do.call(rbind, Xpred_result)) 
  
  Xpred_original[, non_zero_var] = Xpred
  Xpred_original[is.nan(Xpred_original)]=0
  
  return(Xpred_original) 
}   


## ----------------------------------------------------------------------------------------------------------------------------------------------
visualize_digit <- function(missing_X, y, train_removed_rows, per_col, per_row, title){
  par(mfcol=c(per_col, per_row))
  par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i') 
  text(0.5,0.5,"First title",cex=2,font=2)
  for (idx in 1:(per_col*per_row)) { 
      im <- matrix(unlist(missing_X[, ][idx, ]),nrow = 28,byrow = T)
      im <- t(apply(im, 2, rev)) 
      image(1:28, 1:28, im, col=gray((0:255)/255), xaxt='n', main=paste(y[train_removed_rows,,drop=F][idx, ]))
  }
} 


## ---- message = FALSE--------------------------------------------------------------------------------------------------------------------------
impDi_run <- function(X.train, y.train, X.test, y.test, threshold){ 
  #a) on training set 
  X.train[is.na(X.train)] <- NaN
  sigmaDper = dpers(X.train)
  print("S is done")
  
  X_imp.train = impDi(sigmaDper, X.train, threshold)[,, drop=F]
  #b) on testing set   
  print("imp train is done") 
  X.test[is.na(X.test)] <- NaN
  X_imp.test = impDi(sigmaDper, X.test, threshold)[,, drop=F]
  
  result = list("train" = X_imp.train, "test" = X_imp.test)
  return(result)
}  


## ---- message = FALSE--------------------------------------------------------------------------------------------------------------------------
softImpute_run <- function(X.train, y.train, X.test, y.test){
  #a) on training set
  fit_train = softImpute(as.matrix(X.train) , type = 'als') 
  X_imp.train = softImpute::complete(
                              as.matrix(X.train), 
                              fit_train)[,, drop=F]
  #b) on testing set 
  fit_test = softImpute(as.matrix(X.test) , type = 'als') 
  X_imp.test = softImpute::complete(
                            as.matrix(X.test), 
                            fit_test)[,, drop=F]
  result = list("train" = X_imp.train, "test" = X_imp.test )
  return(result)
} 


## ----------------------------------------------------------------------------------------------------------------------------------------------
get_image_position_spatial_to_flatten<- function(delImgPosWidth, delImgPosHeight){ 
  # delImgPosHeight: row 
  # delImgPosWeight : col 
  tmp = c(1:784)
  im <- matrix(unlist(tmp),nrow = 28,byrow = T)
  idxs = im[delImgPosHeight, delImgPosWidth]  
  return(matrix(idxs,nrow = 1,byrow = T)[, ])
}


## ----------------------------------------------------------------------------------------------------------------------------------------------
image_edge_deleting <- function(
    data, 
    delete_type, #by_percent, by_pixel_number 
    percents_of_data,
    image_width,
    image_height,
    width_del_percent=0, 
    height_del_percent=0,
    from_pixel_width=None, 
    from_pixel_height=None
    ){
  if (delete_type =='by_percent'){
    n = dim(data)[2]
    from_pixel_width = ceiling((1-width_del_percent)*image_width)   
    from_pixel_height = ceiling((1-height_del_percent)*image_height)
  }
  if (delete_type=='by_pixel_number'){
    from_pixel_width = from_pixel_width
    from_pixel_height = from_pixel_height
  }
  
flatten_columns_removed = get_image_position_spatial_to_flatten(
  from_pixel_width:image_width,
  from_pixel_height: image_height
)

  flatten_rows_removed = sample.int(nrow(data), as.integer(nrow(data)*percents_of_data))
  missing_data = data
  missing_data[flatten_rows_removed, flatten_columns_removed] <- NA
  result = list(
    'missing_data'=missing_data, 
    'flatten_columns_removed'=flatten_columns_removed,  
    'flatten_rows_removed'=flatten_rows_removed
  ) 
  return(result)
}



## ----------------------------------------------------------------------------------------------------------------------------------------------
# visualize the deleted images 
visualize_digit <- function(missing_X, y, train_removed_rows, per_col, per_row, title){

  par(mfcol=c(per_col, per_row))
  par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')  
 
  for (idx in 1:(per_col*per_row)) { 
      im <- matrix(unlist(missing_X[train_removed_rows, ][idx, ]),nrow = 28,byrow = T)
      im <- t(apply(im, 2, rev)) 
      image(1:28, 1:28, im,  xaxt='n', main=paste(y[train_removed_rows,,drop=F][idx, ])) 
      #image(1:28, 1:28, im, col=gray((0:255)/255), xaxt='n', main=paste(y[train_removed_rows,,drop=F][idx, ]))
  }

} 



## ----------------------------------------------------------------------------------------------------------------------------------------------
#data normalization
sampling_data <- function(data, y_col_name, sample_perc){
  data$label= data[, y_col_name]
  sample_indices <- sample.split(Y=data[, 'label'], SplitRatio = sample_perc)
  sample_data <- as.data.frame(subset(data, sample_indices == TRUE))
  result = list(
    'sample'=sample_data, 
    'X'=as.matrix(sample_data[, 1:(28*28)]), 
    'y'=sample_data[,'label', drop=F]
  )
  return(result)
}



## ----------------------------------------------------------------------------------------------------------------------------------------------
normalizing <- function(x=None, Xtrain=None){
  na_mask = is.na(x)
  mean = apply(Xtrain, 2, mean, na.rm=TRUE)
  sd = apply(Xtrain, 2, sd, na.rm=TRUE)
  
  sd_equal_zero_mask = which(sd==0)
  subtract_mean = sweep(x, 2, mean, '-')
  X_normed = sweep(subtract_mean, 2, sd, "/")
  
  X_normed[is.na(X_normed)] = 0 
  X_normed[is.infinite(X_normed)] = 0 
  X_normed[na_mask] = NA 
  result = list('X_normed'=X_normed, 'mean'=mean, 'sd'=sd, 'sd_equal_zero_mask'=sd_equal_zero_mask)
  return (result) 
}

reconstructingNormedMatrix <- function(X_norm, mean, std){
  mult = sweep(X_norm, 2, std, '*')
  reconstrc = sweep(mult, 2, mean, '+')
  return (reconstrc)
} 


## ----------------------------------------------------------------------------------------------------------------------------------------------
mnistDataPreparation <- function(
    width_del_percent=None, 
    height_del_percent=None, 
    sample_deleted_percent=None, 
    X.train, 
    X.test, 
    y.train, 
    y.test
    ){
  X_train = as.matrix(X.train)
  X_test = as.matrix(X.test)
  y_train = as.matrix(y.train)
  y_test = as.matrix(y.test) 
  
  #cut a piece of image 
  removed_train = image_edge_deleting(
    X_train,
    'by_percent', 
    sample_deleted_percent, 28, 28, 
    width_del_percent=width_del_percent,
    height_del_percent=height_del_percent)
  
  removed_test= image_edge_deleting(
    X_test,
    'by_percent', 
    sample_deleted_percent, 28,28,
    width_del_percent=width_del_percent,
    height_del_percent=height_del_percent)  
  
  train_removed_rows = removed_train$flatten_rows_removed
  test_removed_rows = removed_test$flatten_rows_removed
  train_removed_columns = removed_train$flatten_columns_removed 
  test_removed_columns = removed_test$flatten_columns_removed 
  
  missing.X_train = removed_train$missing_data
  missing.X_test =  removed_test$missing_data  
  # normalization 
  train_normed = normalizing(x=missing.X_train,Xtrain=missing.X_train)
  missing.X_train_normed = train_normed$X_normed
  missing.X_train_mean = train_normed$mean
  missing.X_train_sd = train_normed$sd 
  
  test_normed = normalizing(x=missing.X_test, Xtrain=missing.X_train)
  missing.X_test_normed = test_normed$X_normed
  
  result = list(
    "missing.X_train_normed" = missing.X_train_normed,
    "y_train" = y_train, 
    "missing.X_test_normed" = missing.X_test_normed, 
    "y_test" = y_test, 
    "train_removed_rows" = train_removed_rows, 
    "test_removed_rows" = test_removed_rows, 
    "missing.X_train_mean" = missing.X_train_mean, 
    "missing.X_train_sd" = missing.X_train_sd 
  )
  return(result) 
}



## ----------------------------------------------------------------------------------------------------------------------------------------------
imputationPipeline<- function(
    width_del_percent=None, 
    height_del_percent=None, 
    sample_deleted_percent=None, 
    correlation_threshold=None, 
    X.train, 
    X.test, 
    y.train, 
    y.test 
){
  missingDataCreated = mnistDataPreparation( 
      width_del_percent=width_del_percent, 
      height_del_percent=height_del_percent, 
      sample_deleted_percent=sample_deleted_percent, 
      X.train, 
      X.test, 
      y.train, 
      y.test
  )
  print(paste0(
    "Starting imputation with weight and weight pc ", 
    width_del_percent, 
    "sample deleted percent", 
    sample_deleted_percent, 
    "correlation threshold", 
    correlation_threshold
    )) 
  
  missing.X_train_normed=missingDataCreated$missing.X_train_normed 
  y_train = missingDataCreated$y_train
  
  missing.X_test_normed = missingDataCreated$missing.X_test_normed
  y_test = missingDataCreated$y_test
  
  train_removed_rows = missingDataCreated$train_removed_rows
  test_removed_rows = missingDataCreated$test_removed_rows
  
  missing.X_train_mean = missingDataCreated$missing.X_train_mean
  missing.X_train_sd = missingDataCreated$missing.X_train_sd
  #---------------------------------------------------------------------------------------------------------------  
  START = Sys.time()   
  result_softImpute = softImpute_run(
    missing.X_train_normed, 
    y_train,  
    missing.X_test_normed, 
    y_test
  ) 
   
  softImputeTime = Sys.time() - START 
  print(softImputeTime)
  
  
  softImpute.Xrecon.test = reconstructingNormedMatrix(
    result_softImpute$test, 
    missing.X_train_mean, 
    missing.X_train_sd 
    )  
  
  softImpute.Xrecon.train = reconstructingNormedMatrix(
    result_softImpute$train, 
    missing.X_train_mean, 
    missing.X_train_sd 
  )   
  #---------------------------------------------------------------------------------------------------------------  
  
  START = Sys.time()   
  result_impDi = impDi_run(
    missing.X_train_normed, 
    y_train,  
    missing.X_test_normed, 
    y_test, 
    correlation_threshold  
  ) 
  impDiTime = Sys.time() - START  
  print(impDiTime)
  
  impDi.Xrecon.test = round(reconstructingNormedMatrix(
    result_impDi$test, 
    missing.X_train_mean , 
    missing.X_train_sd 
    ))
  
  impDi.Xrecon.train = round(reconstructingNormedMatrix(
    result_impDi$train, 
    missing.X_train_mean , 
    missing.X_train_sd
    )) 
  curr_dir = getwd() 
  main_dir = '../../data/mnist/imputed'
      
  width_del = toString(as.integer(width_del_percent*100))
  heigh_del =  toString(as.integer(height_del_percent*100)) 
  percent_img_del =  toString(as.integer(sample_deleted_percent*100)) 
  threshold =  toString(as.integer(correlation_threshold*100))  
  
  sub_folder = paste0(
    "threshold_", 
    threshold, 
    "_deletedWidthHeightPc_", 
    width_del, 
    heigh_del, 
    '_noImagePc_', 
    percent_img_del
  )
  
  sub_path = file.path(curr_dir, main_dir, sub_folder)
  if (dir.exists(sub_path)==F){
    dir.create(sub_path)
  }
  print("imputation is done, start saving result")
  # writing result   
  write.csv(X.train, file.path(curr_dir, main_dir,'../processed', "Xtrain.csv"), row.names=FALSE) 
  write.csv(X.test, file.path(curr_dir, main_dir,'../processed', "Xtest.csv"), row.names=FALSE) 
  
  # imputed data  
  write.csv(result_impDi$train, file.path(sub_path, 'train_impDi.csv'), row.names=FALSE) 
  write.csv(result_impDi$test, file.path(sub_path, 'test_impDi.csv'), row.names=FALSE) 
  write.csv(result_softImpute$train, file.path(sub_path, 'train_softImpute.csv'), row.names=FALSE)
  write.csv(result_softImpute$test, file.path(sub_path, 'test_softImpute.csv'), row.names=FALSE) 
  
  #rescaled as original size of image 
  write.csv(impDi.Xrecon.train, file.path(sub_path, 'train_impDi_Xrecon.csv'), row.names=FALSE) 
  write.csv(impDi.Xrecon.test, file.path(sub_path, 'test_impDi_Xrecon.csv'), row.names=FALSE) 
  write.csv(softImpute.Xrecon.train, file.path(sub_path, 'train_softImpute_Xrecon.csv'), row.names=FALSE)  
  write.csv(softImpute.Xrecon.test, file.path(sub_path, 'test_softImpute_Xrecon.csv'), row.names=FALSE)  
  
  #labeled 
  write.csv(y_train, file.path(sub_path, 'y_train.csv'), row.names=FALSE)
  write.csv(y_test, file.path(sub_path, 'y_test.csv'), row.names=FALSE)  
  
  imputation_time <- vector(mode='list', length = 2)
  imputation_time[[1]]<c('softImpute', 'DIMV')
  imputation_time[[2]]<c(softImputeTime, impDiTime)
  exportJson <- toJSON(imputation_time)
  write(exportJson, 'imputation_time.json') 
  
  #saving plot ---------------
  softImputeImgPath <- file.path(sub_path, "softImpute_test.png") 
  png(softImputeImgPath, width=dev.size("px")[1] , height = dev.size("px")[2])  
  visualize_digit(softImpute.Xrecon.test, y_test, test_removed_rows, 2, 6) 
  dev.off()
  
  impDiImgPath <- file.path(sub_path, "impDi_test.png")  
  png(impDiImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
  visualize_digit(impDi.Xrecon.test, y_test, test_removed_rows, 2, 6) 
  dev.off()
  merged_width = dev.size("px")[1]
  merged_height = dev.size("px")[2]*2+50  
  
  
  p1 <- ggdraw() + draw_image(softImputeImgPath)
  p2 <- ggdraw() + draw_image(impDiImgPath) 
  
  imgPath = file.path(curr_dir, main_dir, paste0(sub_folder, '.png'))
  
  png(imgPath,  width = merged_width, height = merged_height) 
  
  plot_grid(p1, p2, 
            nrow = 2, 
            ncol=1,
            labels =  c('SoftImpute', 'impDi'), 
            label_size = 12, 
            scale=1,
            vjust=c(2.0, 2.0),
            label_colour = "blue")
  dev.off()
  # 
  file.remove(softImputeImgPath)
  file.remove(impDiImgPath) 
  print(paste0("Complete saving plot, pipeline is done", sub_folder))
  #done saving plot --------------- 
} 


## ----------------------------------------------------------------------------------------------------------------------------------------------
# width_height_percentages =c(0.4,0.6,0.8)
# sample_deleted_percentages = c(0.2, 0.5)
# correlation_threshold = c(0.3, 0.5)

# width_height_percentages =c(0.5,0.6,0.7) 
# sample_deleted_percentages = c(0.2, 0.5)
# correlation_threshold = c(0.3,0.5,0.7)


width_height_percentages =c(.4, .5, .6, .7) 
sample_deleted_percentages = c(.5)
correlation_threshold = c(.3, .5, .7)  
 
for (width_height_pc in width_height_percentages){
  for (sample_pc in sample_deleted_percentages){
    for (th in correlation_threshold){
      imputationPipeline(
        width_del_percent=width_height_pc,
        height_del_percent=width_height_pc,
        sample_deleted_percent=sample_pc,
        correlation_threshold=th,
        X.train,
        X.test,
        y.train,
        y.test
        )
    }

  }

}



