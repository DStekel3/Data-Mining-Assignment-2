library(tm)
library(RWeka)

# Get trainset as a Corpus
GetTrainset <- function(){
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
  
  # Remove punctuation marks (commas, etc.)
  reviews.trainset.all <- tm_map(reviews.trainset.all,removePunctuation)
  # Make all letters lower case
  reviews.trainset.all <- tm_map(reviews.trainset.all,content_transformer(tolower))
  # Remove excess whitespace
  reviews.trainset.all <- tm_map(reviews.trainset.all,stripWhitespace)
  # Remove numbers
  reviews.trainset.all <- tm_map(reviews.trainset.all,removeNumbers)
  # Remove stopwords
  reviews.trainset.all <- tm_map(reviews.trainset.all, removeWords, stopwords("english"))
  return(reviews.trainset.all)
}

# Get testset as a Corpus
GetTestset <- function(){
  # load in the testset
  reviews.testset.dec <- VCorpus(DirSource("dataset/testset/dec", recursive = TRUE, encoding="UTF-8"))
  
  reviews.testset.tru <- VCorpus(DirSource("dataset/testset/tru", recursive = TRUE, encoding="UTF-8"))
  reviews.testset.all <- c(reviews.testset.dec, reviews.testset.tru)
  
  # Remove punctuation marks (commas, etc.)
  reviews.testset.all <- tm_map(reviews.testset.all,removePunctuation)
  # Make all letters lower case
  reviews.testset.all <- tm_map(reviews.testset.all,content_transformer(tolower))
  # Remove excess whitespace
  reviews.testset.all <- tm_map(reviews.testset.all,stripWhitespace)
  # Remove numbers
  reviews.testset.all <- tm_map(reviews.testset.all,removeNumbers)
  # Remove stopwords
  reviews.testset.all <- tm_map(reviews.testset.all, removeWords, stopwords("english"))
}

BigramTokenizer <- function(x) {NGramTokenizer(x, Weka_control(min = 2, max = 2))}

GetTrainsetBi <- function(){
  
  # join truthful and deceptive data into a single corpus
  trainset <- GetTrainset()
  # create document-term matrix from training corpus
  train.dtm <- DocumentTermMatrix(trainset)
  
  # remove features that occur in less than 2% of the documents
  train.dtm <- removeSparseTerms(train.dtm, GetSparseTermThreshold())
  
  # extract bigrams
  train.dtm2 <- DocumentTermMatrix(trainset,
                                   control = list(tokenize = BigramTokenizer))
  
  # more than 40000 bigrams!
  dim(train.dtm2)
  
  train.dtm2 <- removeSparseTerms(train.dtm2,GetSparseTermThreshold())
  
  dim(train.dtm2)
  
  train.dat1 <- as.matrix(train.dtm)
  
  train.dat2 <- as.matrix(train.dtm2)
  
  # combine unigrams and bigrams leads to 895 features.
  train.dat <- cbind(train.dat1,train.dat2)
  
  return(train.dat)
}

GetSparseTermThreshold <- function(){
  return(0.98)
}