# Anime Recommendations Database – Exploratory Data Analysis

## Project Overview

This project performs Exploratory Data Analysis (EDA) on the Anime Recommendations Database from Kaggle using Python, Pandas, and Matplotlib.

The main objective is to analyze anime ratings, popularity, genres, episode counts, and user preferences through data visualization.

## Dataset

**Source:** [Anime Recommendations Database – Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

The dataset contains two CSV files:

### 1. anime.csv

* `anime_id`: Unique ID of each anime.
* `name`: Name of the anime.
* `genre`: Genres associated with the anime.
* `type`: Type of anime, such as TV, Movie, or OVA.
* `episodes`: Number of episodes.
* `rating`: Average rating out of 10.
* `members`: Number of community members.

### 2. rating.csv

* `user_id`: Unique user identifier.
* `anime_id`: ID of the anime rated.
* `rating`: Rating assigned by a user. A value of -1 indicates that no numerical rating was assigned.

## Objectives

1. Understand the dataset.
2. Identify and handle missing and invalid values.
3. Analyze anime ratings and popularity.
4. Identify the most common anime genres.
5. Compare different anime types.
6. Explore episode-count distributions.
7. Create eight visualizations using Matplotlib.
8. Interpret the patterns observed in the data.

## Graphs Included

1. **Top 10 Anime by Average Rating:** Horizontal bar chart.
2. **Distribution of User Ratings:** Histogram.
3. **Top 10 Most Popular Anime:** Horizontal bar chart.
4. **Top 10 Most Common Anime Genres:** Horizontal bar chart.
5. **Anime Type Distribution:** Pie chart.
6. **Anime Popularity vs Average Rating:** Scatter plot.
7. **Distribution of Anime Episodes:** Histogram.
8. **Anime Ratings by Type:** Box plot.

## Technologies Used

* Python
* Pandas
* Matplotlib

## Analysis and Expected Outcomes

The analysis explores the following questions:

* Which anime have the highest average ratings?
* Which anime have the largest number of members?
* Which genres occur most frequently?
* Which types of anime are most common?
* How are user ratings distributed?
* Is there a relationship between popularity and average rating?
* How are episode counts distributed?
* How do average ratings vary across anime types?

The actual findings will be determined after running the analysis on the dataset.

## Conclusion

This project demonstrates how Python, Pandas, and Matplotlib can be used to perform Exploratory Data Analysis on a real-world anime dataset. The visualizations help identify patterns in anime popularity, ratings, genres, formats, and episode counts.

The analysis can also serve as a foundation for developing an anime recommendation system.

## Dataset Credit

CooperUnion, *Anime Recommendations Database*, Kaggle.

https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
