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
