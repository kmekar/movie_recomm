
library(tidyverse)
library(caret)
library(lubridate)


#############################################################
#### PreProcessing #####
#############################################################

### Import or Download Dataset
if(file.exists("ml-10M100K/ratings.dat")){
  # import movie ratings
  ratings <- read.table(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")),
                        col.names = c("userId", "movieId", "rating", "timestamp"))
  # import movie details
  movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
} else{
  # download movie rating data
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  # unzip/import movie ratings
  ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                        col.names = c("userId", "movieId", "rating", "timestamp"))
  # unzip/import movie details
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
}

### Tidy Data
# remove movie year from title and create column for movie year
pattern <- "^(.*) \\(([0-9 \\-]*)\\)$"
identical(length(str_detect(movies$title, pattern)), nrow(movies)) # check that pattern is detected for all movies
movies <- movies %>%
  mutate(title = str_trim(title)) %>%
  extract(title, c("title_temp", "year"), regex = pattern, remove=F) %>%
  mutate(year = if_else(str_length(year) > 4, as.integer(str_split(year, "-", simplify = T)[1]), as.integer(year))) %>%
  mutate(title = if_else(is.na(title_temp), title, title_temp)) %>%
  select(-title_temp)
sum(is.na(movies$year))

# convert genre string into seperate binary variables (not yet implemented or used)

# combine movie ratings and movie details
movielens <- left_join(ratings, movies, by = "movieId")
head(movielens)

# convert timestamp to datetime and remove timestamp
movielens <- movielens %>% mutate(rate_datetime = as_datetime(timestamp)) %>% 
  select(-timestamp)


### Create training set and test/validation set

# Create test set - 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
train <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

# Remove unnecessary variables
rm(dl, test_index, temp, removed)


####################################
#### Explore Data ##################
####################################

head(train)
dim(train)

## Movies
# number of unique movies
nrow(movies)

# oldest and newest movies
min(movielens$year)
max(movielens$year)

# histogram of movie release year
# hist(movielens$year)
movielens %>%
  ggplot(aes(year)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Movie Release Year") +
  ylab("number of movies")

# movies with the most ratings 
train %>% group_by(title) %>% summarize(n=n()) %>% arrange(desc(n)) %>% head(n=5)

# table of movie genres and their frequency
movies %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% 
  head(n=5) %>% 
  knitr::kable()

# histogram of movie rating values
# hist(train$rating)
movielens %>% 
  ggplot(aes(rating)) + 
  geom_histogram(bins = 10, color = "black") + 
  ggtitle("Movie Ratings")

# histogram of movie rating count
movielens %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movie Rating Count") +
  xlab("number of ratings") +
  ylab("number of movies")

movielens %>% 
  count(movieId) %>% 
  summarise(quantile(n, 0.1))


## Users
# number of unique users
length(unique(train$userId))

# first and last ratings
min(movielens$rate_datetime)
max(movielens$rate_datetime)

# users with the most ratings 
train %>% group_by(userId) %>% summarize(n=n()) %>% arrange(desc(n)) %>% head(n=5)

# histogram of user rating frequency
movielens %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("User Rating Count") +
  xlab("number of ratings") +
  ylab("number of users")


######################################
#### Models and Predictions ##########
######################################

# Define RMSE function to assess model performance
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

### Naive Model
# start with simplest possible model (same rating for all movies for all users):
mu <- mean(train$rating)
mu
naive_rmse <- RMSE(test$rating, mu)
naive_rmse
# off by more than one star

# Create a table to compare different approaches
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

### Model 2 
# predict movie rating by the average rating for that movie
# i.e. include movie effects (b_i) in the naive model
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_2_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",  
                                     RMSE = model_2_rmse))
rmse_results

### Model 3
# add user specific effect (b_u) to model 1
# i.e. account for users who on average rate movies higher or lower than the average
user_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                     RMSE = model_3_rmse))
rmse_results

### Model 4 
# use regularization to account for movies with a low number of ratings 
# and users with fewer ratings

# use cross validation to pick lambda, the regularization penalty parameter
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train$rating)
  
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test$rating))
})

qplot(lambdas, rmses) 
lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable(digits = 5)

