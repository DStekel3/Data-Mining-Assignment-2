# load the tm package
library(tm)
# Read in the trainset data using UTF-8 encoding
reviews.dec <- VCorpus(DirSource("dataset/trainset/dec", recursive = TRUE,
                                  encoding="UTF-8"))
reviews.tru <- VCorpus(DirSource("dataset/trainset/tru", recursive = TRUE,
                                  encoding="UTF-8"))


# join truthful and deceptive data into a single corpus
reviews.trainset.all <- c(reviews.dec, reviews.tru)

# create label vector (0=deceptive, 1=truthful)
reviews.trainset.labels <- c(rep(0,320), rep(1,320))

# PRE-PROCESSING

# Remove punctuation marks (comma’s, etc.)
reviews.trainset.all <- tm_map(reviews.trainset.all,removePunctuation)
# Make all letters lower case
reviews.trainset.all <- tm_map(reviews.trainset.all,content_transformer(tolower))
# Remove stopwords
reviews.trainset.all <- tm_map(reviews.trainset.all, removeWords,
                        stopwords("english"))
# Remove numbers
reviews.trainset.all <- tm_map(reviews.trainset.all,removeNumbers)
# Remove excess whitespace
reviews.trainset.all <- tm_map(reviews.trainset.all,stripWhitespace)

# indexes of truthful and deceptive documents as trainset
index.tru <- 1:320
index.dec <- 321:640
index.train <- c(index.tru, index.dec)

# create document-term matrix from training corpus
train.dtm <- DocumentTermMatrix(reviews.trainset.all[index.train])

# remove feature that occus in less than 5% of the documents
train.dtm <- removeSparseTerms(train.dtm,0.95)

# train the naive bayes multinomial classifier using the trainset
reviews.mnb <- train.mnb(as.matrix(train.dtm),reviews.trainset.labels[index.train])

# load in the testset
reviews.testset.dec <- VCorpus(DirSource("dataset/testset/dec", recursive = TRUE,
                                     encoding="UTF-8"))

reviews.testset.tru <- VCorpus(DirSource("dataset/testset/tru", recursive = TRUE,
                                         encoding="UTF-8"))

reviews.testset.all <- c(reviews.testset.dec, reviews.testset.tru)

reviews.testset.labels <- c(rep(0, 80), rep(1,80))

# create document term matrix for test set
test.dtm <- DocumentTermMatrix(reviews.testset.all, list(dictionary=dimnames(train.dtm)[[2]]))

reviews.mnb.pred <- predict.mnb(reviews.mnb,as.matrix(test.dtm))

print(table(reviews.mnb.pred,reviews.testset.labels))

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

