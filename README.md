# Anime Data Analysis & Recommendation System

Exploratory data analysis and three recommendation models built on the
[Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
(MyAnimeList data from Kaggle).

## What this project does

1. Loads and cleans `anime.csv` and `rating.csv`.
2. Saves the cleaned anime table to `anime_clean.csv`.
3. Creates 12 visualizations in `images/`.
4. Builds three recommenders: popularity, content-based and item-based collaborative filtering.
5. Optionally evaluates collaborative filtering against a popularity baseline (hit rate).

## Dataset

Download the two files from Kaggle and place them in the project folder (or pass `--data-dir`).
`rating.csv` is large, so it is not stored in this repository.

**anime.csv**

| Column | Meaning |
|---|---|
| `anime_id` | Unique ID of each anime |
| `name` | Name of the anime |
| `genre` | Comma-separated genres |
| `type` | TV, Movie, OVA, Special, ONA or Music |
| `episodes` | Number of episodes (`Unknown` becomes missing) |
| `rating` | Average rating out of 10 |
| `members` | Number of community members |

**rating.csv**

| Column | Meaning |
|---|---|
| `user_id` | Unique user identifier |
| `anime_id` | ID of the anime rated |
| `rating` | User rating 1-10; `-1` means watched but not scored (removed during cleaning) |

## Cleaning steps

- HTML entities in names are decoded (for example `&#039;`).
- `episodes` is converted to a number; `Unknown` becomes missing.
- Missing `genre` is filled with an empty string and missing `type` with `Unknown`.
- Missing `rating` values are kept as missing and excluded from rating statistics.
- Rows with `rating = -1` in `rating.csv` are removed.

## Visualizations (12)

All charts are produced by `anime_analysis.py` and saved to `images/`.

| # | File | Chart |
|---|---|---|
| 01 | `01_missing_values.png` | Missing values in anime.csv (bar) |
| 02 | `02_type_distribution.png` | Number of anime by type (bar) |
| 03 | `03_avg_rating_distribution.png` | Distribution of average anime rating (histogram) |
| 04 | `04_user_rating_distribution.png` | Individual user ratings, 1-10 (bar) |
| 05 | `05_top_genres.png` | Top 15 genres by number of anime (bar) |
| 06 | `06_genre_avg_rating.png` | Highest-rated genres with at least 100 anime (bar) |
| 07 | `07_rating_by_type.png` | Average rating by anime type (box plot) |
| 08 | `08_members_vs_rating.png` | Popularity vs rating, with Spearman correlation (scatter) |
| 09 | `09_episodes_distribution.png` | Episode count, up to 100 episodes (histogram) |
| 10 | `10_top_popular.png` | Top 10 most popular anime by members (bar) |
| 11 | `11_user_item_activity.png` | Ratings per user and per anime, long tail (histograms) |
| 12 | `12_genre_correlation.png` | Genre co-occurrence correlation (heatmap) |

## Recommenders

- **Popularity:** IMDB-style weighted rating, `WR = v/(v+m)*R + m/(v+m)*C`, computed for anime in the top 20% by members.
- **Content-based:** TF-IDF on genre and type, ranked by cosine similarity. Ties go to the more popular anime.
- **Item-based collaborative filtering:** cosine similarity between items on mean-centred user ratings, using anime with at least 50 ratings.

**Evaluation (`--evaluate`):** users are split 80/20. For each test user, one liked anime (rating of 8 or more) is hidden and the model must recover it in its top-10 list. Results are compared with a popularity baseline.

## How to run

```bash
pip install -r requirements.txt

python anime_analysis.py                          # clean + plots + sample recommendations
python anime_analysis.py --data-dir path/to/csvs  # where anime.csv and rating.csv live
python anime_analysis.py --title "Death Note"     # recommendations for another title
python anime_analysis.py --evaluate               # adds offline evaluation (takes several minutes)
```

The title passed to `--title` must match a name in `anime.csv` exactly. Collaborative filtering also needs the anime to have at least 50 ratings.

## Project files

| File | Purpose |
|---|---|
| `anime_analysis.py` | Cleaning, 12 plots, recommenders and evaluation |
| `project.ipynb` | Notebook version of the project |
| `anime_clean.csv` | Cleaned anime table with per-anime rating statistics |
| `insights_report.md` | Findings, recommender results and conclusions |
| `requirements.txt` | Python dependencies |

## Technologies

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, seaborn.

## Dataset credit

CooperUnion, *Anime Recommendations Database*, Kaggle.
https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
