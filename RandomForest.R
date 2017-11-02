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

# remove features that occur in less than 2% of the documents
train.dtm <- removeSparseTerms(train.dtm,0.98)

# include bigrams
train.dtm <- GetTrainsetBi()

matrix <- as.matrix(train.dtm)
labels <- reviews.trainset.labels
matrix <- cbind(matrix, labels)
trainset.frame <- as.data.frame(matrix)
#names(trainset.frame)[names(trainset.frame ) == 'next'] <- 'next_term'
#names(trainset.frame)[names(trainset.frame ) == 'else'] <- 'else_term'
#names(trainset.frame)[names(trainset.frame ) == 'break'] <- 'break_term'
names(trainset.frame) <- make.names(names(trainset.frame))
rownames(trainset.frame) <- c()

forest <- randomForest(formula = as.factor(labels) ~., data = trainset.frame, ntree=100, replace=TRUE)

# load in the testset
reviews.testset.all <- GetTestset()
testset.labels <- c(rep(0, 80), rep(1,80))
test.dtm <- DocumentTermMatrix(reviews.testset.all, list(dictionary=dimnames(train.dtm)[[2]]))

testset.frame <- as.data.frame(as.matrix(test.dtm), stringsAsFactors = FALSE)
#names(testset.frame)[names(testset.frame) == 'next'] <- 'next_term'
#names(testset.frame)[names(testset.frame) == 'else'] <- 'else_term'
#names(testset.frame)[names(testset.frame) == 'break'] <- 'break_term'
names(trainset.frame) <- make.names(names(trainset.frame))
rownames(trainset.frame) <- c()

prediction <- predict(forest, testset.frame, type = "class")

print(table(prediction, testset.labels))