library(rpart)

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

matrix <- as.matrix(train.dtm)
label <- reviews.trainset.labels

tree <- rpart(label ~.,data=as.data.frame(cbind(matrix, label)),cp=0,minbucket=1,minsplit=2,method="class")
printcp(tree)

# get pruned tree, based on the best value for cp
tree.pruned <- prune(tree,cp=0.12)
plot(tree.pruned)