source('DataLoader.R')

library(rpart)

# get the trainset
trainset <- GetTrainset()

# create label vector (0=deceptive, 1=truthful)
trainset.labels <- c(rep(0,320), rep(1,320))

# create document-term matrix from training corpus
train.dtm <- DocumentTermMatrix(trainset)


# remove features that occur in less than 2% of the documents
train.dtm <- removeSparseTerms(train.dtm,0.98)

# include bigrams
train.dtm <- GetTrainsetBi()

matrix <- as.matrix(train.dtm)
label <- trainset.labels

tree <- rpart(label ~.,data=as.data.frame(cbind(matrix, label)),cp=0,minbucket=1,minsplit=2,method="class")
print(tree$cptable)
plotcp(tree)

# get pruned tree, based on the best value for cp
tree.pruned <- prune(tree,cp=0.1)

# load in the testset
testset <- GetTestset()

# set up labels
testset.labels <- c(rep(0, 80), rep(1,80))

# create document term matrix for test set
test.dtm <- DocumentTermMatrix(testset, list(dictionary=dimnames(train.dtm)[[2]]))

prediction <- predict(tree.pruned, as.data.frame(as.matrix(test.dtm)), type = "class")

# show confusion matrix
print(table(prediction, testset.labels))