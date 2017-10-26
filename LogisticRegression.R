source('DataLoader.R')

# load the tm package
library(RWeka)
library(tm)
library(glmnet)
BigramTokenizer <- function(x) {NGramTokenizer(x, Weka_control(min = 2, max = 2))}

# get the trainset
trainset <- GetTrainset()

# create label vector (0=deceptive, 1=truthful)
trainset.labels <- c(rep(0, 320), rep(1, 320))

# create document-term matrix from training corpus
train.dtm <- DocumentTermMatrix(trainset)

# remove feature that occus in less than 5% of the documents
train.dtm <- removeSparseTerms(train.dtm, 0.95)

testset <- GetTestset()

# create document term matrix for test set
test.dtm <-
  DocumentTermMatrix(testset, list(dictionary = dimnames(train.dtm)[[2]]))

# set up labels
testset.labels <- c(rep(0, 80), rep(1, 80))

# logistic regression with lasso penalty
reviews.glmnet <-
  cv.glmnet(
    as.matrix(train.dtm),
    trainset.labels,
    family = "binomial",
    type.measure = "class"
  )


# make predictions on the test set
reviews.logreg.pred <- predict(
  reviews.glmnet,
  newx = as.matrix(test.dtm),
  s = "lambda.1se",
  type = "class"
)
# show confusion matrix (basic logistic regression)
print("Basic Logistic regression:")
print(table(reviews.logreg.pred, testset.labels))

UsingTermFrequencies(trainset, trainset.labels, testset, testset.labels)
UsingBigrams(trainset, trainset.labels, testset, testset.labels)

UsingTermFrequencies <- function(trainset, trainset.labels, testset, testset.labels){
  # construct training document term matrix with tf-idf weights
  train2.dtm <- DocumentTermMatrix(trainset,
                                   control = list(weighting = weightTfIdf))
  # remove sparse terms
  train2.dtm <- removeSparseTerms(train2.dtm, 0.95)
  
  # create document term matrix for test set
  test.dtm <-
    DocumentTermMatrix(testset, list(dictionary = dimnames(train.dtm)[[2]]))
  
  # perform logistic regression with lasso penalty
  reviews2.glmnet <-
    cv.glmnet(
      as.matrix(train2.dtm),
      trainset.labels,
      family = "binomial",
      type.measure = "class"
    )
  
  # compute tf-idf scores on test set: we use the document frequency
  # from the training set!
  train2.dtm <- as.matrix(train.dtm)
  
  # convert term frequency counts to binary indicator
  train2.dtm <- matrix(as.numeric(train2.dtm > 0), nrow = 640, ncol = 321)
  
  # sum the columns of the training set
  train2.idf <- apply(train2.dtm, 2, sum)
  # compute idf for each term (column)
  train2.idf <- log2(640 / train2.idf)
  # term frequencies on the test set
  test2.dtm <- as.matrix(test.dtm)
  # compute tf-idf weights on the test set
  for (i in 1:321)
  {
    test2.dtm[, i] <- test2.dtm[, i] * train2.idf[i]
  }
  
  # make predictions on the test set using lambda=lambda.1se
  reviews.logreg2.pred <- predict(reviews2.glmnet,
                                  newx = test2.dtm,
                                  s = "lambda.1se",
                                  type = "class")
  
  print("Logistic regression using td-idf weights:")
  
  # show confusion matrix using tf-idf weights
  print(table(reviews.logreg2.pred, testset.labels))
}

UsingBigrams <- function(trainset, trainset.labels, testset, testset.labels){
  # extract bigrams
  train.dtm2 <- DocumentTermMatrix(trainset,
                                   control = list(tokenize = BigramTokenizer))
  
  # more than 40000 bigrams!
  dim(train.dtm2)
  
  train.dtm2 <- removeSparseTerms(train.dtm2,0.99)
  
  # only 461 left :)
  dim(train.dtm2)
  
  train.dat1 <- as.matrix(train.dtm)
  
  train.dat2 <- as.matrix(train.dtm2)
  
  # combine unigrams and bigrams leads to 782 features.
  train.dat <- cbind(train.dat1,train.dat2)
  
  dim(train.dat)
  
  # fit regularized logistic regression model
  # use cross-validation to evaluate different lambda values
  reviews3.glmnet <- cv.glmnet(train.dat, trainset.labels,
                               family="binomial",type.measure="class")
  
  # show coefficient estimates for lambda-1se
  # (only a selection of the bigram coefficients is shown here)
  coef(reviews3.glmnet,s="lambda.1se")
  
  # create document term matrix for the test data,
  # using the training dictionary
  test3.dtm <- DocumentTermMatrix(testset,
                                  list(dictionary=dimnames(train.dat)[[2]]))
  
  # convert to ordinary matrix
  test3.dat <- as.matrix(test3.dtm)
  
  # get columns in the same order as on the training set
  test3.dat <- test3.dat[,dimnames(train.dat)[[2]]]
  
  # make predictions using lambda.1se
  reviews.logreg3.pred <- predict(reviews3.glmnet,newx=test3.dat,
                                  s="lambda.1se",type="class")
  
  print("Logistic regression using bigrams:")
  print(table(reviews.logreg3.pred,testset.labels))
}

