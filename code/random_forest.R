rm(list=ls())


library(caret)
library(randomForest)
library(tidyverse)

# these paths need to be changed to correspond to where the data is located on your system
x_file <- 'data/MS_features1.csv'
y_file <- 'data/bioactivity.csv'

# change this to the location you want to save the importance values to
save_name <- 'results/first_stage_random_forest_scrambled_importance.csv'

# data "backwards"? (p rows, n columns)
is_backwards = FALSE

# scramble input/output relationships? (gives a sort of null importance)
# set this to FALSE to do the actual analysis
scramble = TRUE

# log transform X and Y?
log_trans = TRUE

# input data to glmnet needs to be a matrix of n x p, and a vector (y) of n values
#  read data and try to cram it into the correct format

x <- read.table(x_file,header=FALSE,sep=",")
x <- as.matrix(x)

if (is_backwards) {
  colnames(x) <- paste('X',1:dim(x)[2],sep="")
  rownames(x) <- paste('V',1:dim(x)[1],sep="")
  x <- t(x)
} else {
  colnames(x) <- paste('V',1:dim(x)[2],sep="")
  rownames(x) <- paste('X',1:dim(x)[1],sep="")
}

y <- scan(y_file)

# data transformation
if (log_trans) {
  y <- log(y)
  x <- log(1 + x)
}

if (scramble) {
  y <- sample(y,length(y))
}

# make the input data frame
alldata.df <- data.frame(y,x)

# how many runs to aggregate over?
n_mod <- 1000
# how many models used 0 parameters (i.e., just fit an intercept?)
bad_mod <- 0
# accumulated total weights (exp(-Cost))
w_tot <- 0.0
# mean R2
mean_R2 <- 0.0

for (i in 1:n_mod){

  message(sprintf("Current model: %d/%d", i, n_mod))

  # 2/3-1/3 train-test split
  train_rows <- sample(1:dim(x)[1], 0.66*dim(x)[1])

  traindata.df <- alldata.df[train_rows,]
  testdata.df <- alldata.df[-train_rows,]

  # kludge-y way to get x and y
  x.train <- x[train_rows,]
  y.train <- y[train_rows]
  x.test <- x[-train_rows,]
  y.test <- y[-train_rows]

  # determine the number of trees in the forest
  ntree.oob <- double(2000)
  for (n in 1:10) {
    train.rf <- randomForest(y~.,data=traindata.df,ntree=1000)
    ntree.oob <- ntree.oob + train.rf$mse
  }
  ntree.oob <- ntree.oob/10
  ntrees <- which.min(ntree.oob)

  # with a known number of trees, tune mtry
  model.rf <- train(
    x.train, y.train, method="rf", trControl = trainControl("LOOCV",number=3,search="random"),
    ntree = ntrees,tuneLength = 50, importance = TRUE
  )

  # this will predict the holdout data using the model
  predictions.rf <- model.rf %>% predict(x.test)

  # this makes a df that contains the predicted MSE and R2
  df <- data.frame(RMSE.rf = RMSE(predictions.rf, y.test),
         Rsquare.rf = R2(predictions.rf, y.test))

  w <- exp(-1.0*length(y.test)*(df$RMSE.rf)**2)

  if (!is.na(df$Rsquare.rf)) {
    # calculate variable importance
    imp <- varImp(model.rf$finalModel)
    # replace NaNs with zero
    imp[is.na(imp)] = 0.0

    if (!exists("imp_vec")){

      imp_vec <- imp*w

    } else {
      imp_vec <- imp_vec + imp*w
    }
    mean_R2 <- mean_R2 + w*df$Rsquare.rf
    w_tot <- w_tot + w

  } else {
    bad_mod <- bad_mod + 1
  }

}

# make a data frame out of the importance vector
imp_vec <- imp_vec/w_tot
var.imp <- data.frame("var" = row.names(imp_vec), "imp" = imp_vec$Overall)
# sort in variable order
var.imp <- var.imp[order(-var.imp$imp),]

mean_R2 <- mean_R2/w_tot

write.csv(var.imp,file=save_name,quote=FALSE,row.names=FALSE,col.names=FALSE)
