library(ggplot2)
d <- read.csv("song_popularity.csv")
ggplot(d, aes(x=NUMBER_OF_USERS)) + geom_histogram(binwidth=500)
