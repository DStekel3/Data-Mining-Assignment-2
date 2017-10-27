library(tm)

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
  # Remove stopwords
  reviews.trainset.all <- tm_map(reviews.trainset.all, removeWords,
                                 stopwords("english"))
  # Remove numbers
  reviews.trainset.all <- tm_map(reviews.trainset.all,removeNumbers)
  # Remove excess whitespace
  reviews.trainset.all <- tm_map(reviews.trainset.all,stripWhitespace)
  
  reviews.trainset.all <- tm_map(reviews.trainset.all, stemDocument)
  
  reviews.trainset.all <- tm_map(reviews.trainset.all, PlainTextDocument)
  return(reviews.trainset.all)
}

# Get testset as a Corpus
GetTestset <- function(){
  # load in the testset
  reviews.testset.dec <- VCorpus(DirSource("dataset/testset/dec", recursive = TRUE,
                                           encoding="UTF-8"))
  
  reviews.testset.tru <- VCorpus(DirSource("dataset/testset/tru", recursive = TRUE,
                                           encoding="UTF-8"))
  
  reviews.testset.all <- c(reviews.testset.dec, reviews.testset.tru)
  
  # Remove punctuation marks (commas, etc.)
  reviews.testset.all <- tm_map(reviews.testset.all,removePunctuation)
  # Make all letters lower case
  reviews.testset.all <- tm_map(reviews.testset.all,content_transformer(tolower))
  # Remove stopwords
  reviews.testset.all <- tm_map(reviews.testset.all, removeWords,
                                stopwords("english"))
  # Remove numbers
  reviews.testset.all <- tm_map(reviews.testset.all,removeNumbers)
  # Remove excess whitespace
  reviews.testset.all <- tm_map(reviews.testset.all,stripWhitespace)
  reviews.testset.all <- tm_map(reviews.testset.all, PlainTextDocument)
}