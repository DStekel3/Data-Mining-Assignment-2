source('DataLoader.R')


# load the tm package
library(tm)

# Naive Bayes training function
train.mnb <- function (dtm,labels)
{
  call <- match.call()
  V <- ncol(dtm)
  N <- nrow(dtm)
  prior <- table(labels)/N
  labelnames <- names(prior)
  nclass <- length(prior)
  cond.probs <- matrix(nrow=V,ncol=nclass)
  dimnames(cond.probs)[[1]] <- dimnames(dtm)[[2]]
  dimnames(cond.probs)[[2]] <- labelnames
  index <- list(length=nclass)
  for(j in 1:nclass){
    index[[j]] <- c(1:N)[labels == labelnames[j]]
  }
  for(i in 1:V){
    for(j in 1:nclass){
      cond.probs[i,j] <- (sum(dtm[index[[j]],i])+1)/(sum(dtm[index[[j]],])+V)
    }
  }
  list(call=call,prior=prior,cond.probs=cond.probs)
}

# Naive Bayes prediction function
predict.mnb <- function (model,dtm)
{
  classlabels <- dimnames(model$cond.probs)[[2]]
  logprobs <- dtm %*% log(model$cond.probs)
  N <- nrow(dtm)
  nclass <- ncol(model$cond.probs)
  logprobs <- logprobs+matrix(nrow=N,ncol=nclass,log(model$prior),byrow=T)
  classlabels[max.col(logprobs)]
}
# Start the clock!
ptm <- proc.time()
trainset <- GetTrainset()

# create label vector (0=deceptive, 1=truthful)
trainset.labels <- c(rep(0,320), rep(1,320))

# create document-term matrix from training corpus
train.dtm <- DocumentTermMatrix(trainset)

# remove features that occur in less than 2% of the documents
# train.dtm <- removeSparseTerms(train.dtm,GetSparseTermThreshold())

# uncomment the next line for bigrams
# train.dtm <- GetTrainsetBi()

# train the naive bayes multinomial classifier using the trainset
reviews.mnb <- train.mnb(as.matrix(train.dtm),trainset.labels)

testset <- GetTestset()


testset.labels <- c(rep(0, 80), rep(1,80))

# create document term matrix for test set
test.dtm <- DocumentTermMatrix(testset, list(dictionary=dimnames(train.dtm)[[2]]))

# uncomment the next line for bigrams
# test.dtm <- GetTestsetBi()

reviews.mnb.pred <- predict.mnb(reviews.mnb,as.matrix(test.dtm))
# Stop the clock
print(proc.time() - ptm)
print(table(reviews.mnb.pred,testset.labels))

