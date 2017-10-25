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

# logistic regression with lasso penalty
reviews.glmnet <- cv.glmnet(as.matrix(train.dtm),reviews.trainset.labels[index.train],
                              family="binomial",type.measure="class")

# load in the testset
reviews.testset.dec <- VCorpus(DirSource("dataset/testset/dec", recursive = TRUE,
                                         encoding="UTF-8"))

reviews.testset.tru <- VCorpus(DirSource("dataset/testset/tru", recursive = TRUE,
                                         encoding="UTF-8"))

reviews.testset.all <- c(reviews.testset.dec, reviews.testset.tru)

reviews.testset.labels <- c(rep(0, 80), rep(1,80))

# create document term matrix for test set
test.dtm <- DocumentTermMatrix(reviews.testset.all, list(dictionary=dimnames(train.dtm)[[2]]))


# make predictions on the test set
reviews.logreg.pred <- predict(reviews.glmnet,
                               newx=as.matrix(test.dtm),s="lambda.1se",type="class")
# show confusion matrix
print(table(reviews.logreg.pred, reviews.testset.labels))