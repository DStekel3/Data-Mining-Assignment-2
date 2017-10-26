source('DataLoader.R')

library(randomForest)
set.seed(415)

# get the trainset
reviews.trainset.all <- GetTrainset()

# create label vector (0=deceptive, 1=truthful)
reviews.trainset.labels <- c(rep(0,320), rep(1,320))

# indexes of truthful and deceptive documents as trainset
index.tru <- 1:320
index.dec <- 321:640
index.train <- c(index.tru, index.dec)

# create document-term matrix from training corpus
train.dtm <- DocumentTermMatrix(reviews.trainset.all[index.train])

# remove feature that occus in less than 5% of the documents
train.dtm <- removeSparseTerms(train.dtm,0.95)

matrix <- as.matrix(train.dtm)
label <- reviews.trainset.labels
trainset.frame <- as.data.frame(cbind(matrix, label))

forest <- randomForest(as.factor(label) ~., data = trainset.frame, ntree=100, replace=TRUE)

# load in the testset
reviews.testset.all <- GetTestset()