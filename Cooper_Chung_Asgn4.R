# BTMA 531 Assignment 4
# Cooper Chung (Redacted Student ID)
# cooper.chung1@ucalgary.ca

# Question 1a
prsa <- read.csv("PRSA_DataSample.csv", header = T) # Read the data into R

class(prsa)                                   # Check the class first to see if it is time-series - it isn't

prsa_temp_ts <- ts(prsa$TEMP, frequency = 12) # Convert to time-series data with monthly data

library(forecast)                             # Load forecast library

prsa_temp_ts <- na.interp(prsa_temp_ts)       # Interpolate missing values in the dataset using na.interp


# Question 1b
plot(prsa_temp_ts, ylab = "Temperature")      # Plot the time-series data for temperature

library(TTR)                                  # Load TTR library

plot(runMean(prsa_temp_ts, 2))                # Plot moving average time-series data using the moving average of the last 2 observations


# Question 1c
acf(prsa_temp_ts)   # Plot the ACF for our data

pacf(prsa_temp_ts)  # Plot the PACF for our data

# From these two plots, we can see that our data has some autocorrelation. In the ACF, we can see that the data often spikes to outside
# our upper and lower control limits. Then comparing it to the PACF, we can see that a majority of the data now no longer spikes outside
# of the control limits, but can occasionally do so.

library(aTSA)           # Load aTSA package to test for stationary

adf.test(prsa_temp_ts)  # Perform stationary test. We can see that Type 2 is the most impressive, suggesting that data has drift but
# no trend. However, we can see that not all of the p-values are below 0.05, suggesting that the data is NOT stationary.
# Therefore, we must difference the data.


# Question 1d
arima1 <- auto.arima(prsa_temp_ts)  # Use auto.arima to create the model

summary(arima1)             # From the output, we can see that for the MA (Moving Average) model, it used a (p, d, q) of (0, 1, 1).
# p = 0 makes sense since this is the moving average model, but a d value of 1 indicates that the model was differenced.
# a q value of 1 also makes sense since this is the moving average model. We can also see that for the AR model it
# used a (p, d, q) of (0, 1, 1). It actually used a seasonal model with a p value of 1, which makes sense because
# it is the auto regressive model. It also used a d value of 1, indicating that this model is also differenced.
# This model also has a q value of 1 which makes sense since this is the auto regressive model.

arima_pred <- predict(arima1, 2 * 12) # Make the prediction for 24 months, which is just the same as 12 months * 2

ts.plot(prsa_temp_ts, arima_pred$pred, lty = c(1, 3)) # Plot the dataset, along with the prediction.


# Question 1e
plot(decompose(prsa_temp_ts)) # Decompose the time-series data into its components

holt_fit <- HoltWinters(prsa_temp_ts) # Create model via Holt Winters

holt_pred <- forecast::forecast(holt_fit, h = 24) # Predict the next 24 months using the Holt Winters model

plot(holt_pred) # Plot the prediction


# Question 1f
prsa_wspm_ts <- ts(prsa$WSPM, frequency = 12) # Convert to time-series data with monthly data

prsa_wspm_ts <- na.interp(prsa_wspm_ts)       # Interpolate missing values in the dataset using na.interp

fit_temp_wspm <- auto.arima(prsa_wspm_ts, xreg = prsa_temp_ts)  # Use auto.arima to create the regression

library(lmtest) # Load library to help us calculate p-values

coeftest(fit_temp_wspm) # From this, we can see that the coefficient for xreg is 0.390238, which is greater than our 95% significance level (0.05).
# As a result, we can say that the temperature does NOT have a significant impact on the max wind speed.


# Question 3a
library(tm) # Load tm library for text mining

text_file <- read.csv("TextData.csv", header = T)       # Read the csv into R

text <- VCorpus(DataframeSource(text_file))             # Change the format

text <- tm_map(text, content_transformer(tolower))      # Make it all lowercase

text <- tm_map(text, stripWhitespace)                   # Remove all white spaces

text <- tm_map(text, removeWords, stopwords("english")) # Remove all stopwords

text <- tm_map(text, removeNumbers)                     # Remove all numbers

text <- tm_map(text, stemDocument)                      # Stem document


# Question 3b
dtm <- DocumentTermMatrix(text) # Create document term matrix

findFreqTerms(dtm, 20)          # Looking at the most frequent terms used at least 20 times, I will choose the terms "shure" and "guitar"

findAssocs(dtm, "shure", 0.5)   # Find highly associated terms to the two words I mentioned before, with a correlation of 0.5 or more

findAssocs(dtm, "guitar", 0.5)


# Question 3c
tdm <- TermDocumentMatrix(text) # Create term document matrix

library(wordcloud)  # Load library to create wordclouds

wordcloud_matrix <- as.matrix(tdm)  # Turn the term document matrix into an actual matrix recognized in R

wordcloud_frequency <- sort(rowSums(wordcloud_matrix), decreasing = T)  # Turn this matrix into a numerical list where each word is ordered decreasing

set.seed(1) # Set seed for replicability before generating the wordcloud

wordcloud(words = names(wordcloud_frequency), freq = wordcloud_frequency, min.freq = 20, random.order = F)  # Create the wordcloud


# Question 3d
tdm_no_sparse <- removeSparseTerms(tdm, sparse = 0.95)  # Remove sparse terms from our term document matrix, 95%

hier_matrix <- as.matrix(tdm_no_sparse) # Create a new matrix to be used in hierarchical clustering

dist_matrix <- dist(scale(hier_matrix)) # Get distances for clustering based off scaled data

hier_fit <- hclust(dist_matrix) # Cluster terms

plot(hier_fit)  # Plot cluster dendrogram

rect.hclust(hier_fit, k = 5)  # Create 5 clusters


# Question 3e
groups <- cutree(hier_fit, k = 5) # Cut the tree into 5 groups

transposed <- t(hier_matrix)  # Transpose dataset so we have documents instead of terms

set.seed(1) # Set seed for replicability before doing kmeans

kmeans_result <- kmeans(transposed, 5)  # Perform kmeans with k = 5


# Question 3f
library(syuzhet)  # Load syuzhet library

sentiment_list <- get_nrc_sentiment(text_file$text) # Get list of sentiments

sentiment_sums <- colSums(sentiment_list) # Tally the number of occurrences per sentiment

library(ggplot2)  # Load library to bar plot

barplot(sentiment_sums, las = 2)  # Bar plot the sentiments and frequencies