# Insights Report — Anime Recommendation System & Data Analysis

Data: MyAnimeList via Kaggle — 12,294 anime, 6,337,241 explicit user ratings (1-10) from 69,600 users
(7.8 M raw rows; 1.48 M "watched, not scored" `-1` rows removed).

## 1. Data quality
| Column | Issue | Handling |
|---|---|---|
| `rating` | 1.9 % missing | kept as NaN, excluded from rating stats |
| `genre` | 0.5 % missing | filled with empty string |
| `type` | 0.2 % missing | filled with `Unknown` |
| `episodes` | 2.8 % are `"Unknown"` | converted to NaN / numeric |
| `name` | HTML entities (e.g. `&#039;`) | decoded |
| `rating.csv` | `-1` = no score | rows removed |

## 2. Key findings from the visualizations
1. **Catalogue mix** — TV (3,787) and OVA (3,311) dominate; Movies 2,348 (`images/02`).
2. **Ratings are generous** — average anime score is **6.47**, but individual users most often give 7-9; the mean user score is **7.81** (`03`, `04`).
3. **Genres** — Comedy (4,645), Action (2,845), Adventure (2,348), Fantasy (2,309), Sci-Fi (2,070) are most common (`05`).
4. **Quality by genre** — Mystery (7.23) and Police (7.12) are the best-rated large genres; Dementia (5.01) and Music (5.92) the lowest (`06`).
5. **Quality by type** — TV has the highest median rating (6.94); Music (5.63) and ONA (5.76) the lowest (`07`).
6. **Popularity ↔ quality** — Spearman correlation between members and rating is ≈ **0.67**: widely watched anime are usually well rated (`08`).
7. **Short content** — the median anime has just **2 episodes** (OVAs, specials, movies) (`09`).
8. **Most popular** — Death Note, Shingeki no Kyojin, Sword Art Online, Fullmetal Alchemist: Brotherhood (`10`).
9. **Sparse, long-tailed data** — most users rate few titles and a handful of anime collect tens of thousands of ratings; this is the main challenge for recommenders (`11`).
10. **Genre structure** — Sci-Fi ↔ Mecha and Action ↔ Sci-Fi/Shounen co-occur strongly; Comedy is negatively correlated with Drama and Hentai (`12`).

## 3. Recommender results
| Model | How it works | Example output |
|---|---|---|
| Popularity | weighted rating (rating × members) | Fullmetal Alchemist: Brotherhood, Kimi no Na wa., Steins;Gate |
| Content-based | TF-IDF of genre + type, cosine similarity | *Death Note* → Mousou Dairinin, Death Note Rewrite, Higurashi Kai |
| Item-based CF | mean-centred user ratings, cosine between items | *Death Note* → Code Geass (+R2), FMA: Brotherhood, Steins;Gate, Shingeki no Kyojin |

**Offline evaluation** (`--evaluate`, 1,500 unseen users, one liked anime hidden, top-10 list):

| Model | HitRate@10 |
|---|---|
| Popularity baseline | 11.3 % |
| **Item-based CF** | **22.5 %** |

## 4. Conclusions
- Collaborative filtering recovers hidden favourites **twice as often** as recommending what is popular.
- Content-based recommendations are explainable ("same genres") and work for new anime that have no ratings yet, but ignore taste patterns that cross genres (CF links *Death Note* to *Code Geass*, which share almost no genres).
- Best practical system: **hybrid** — CF for titles with enough ratings, content-based for cold-start, popularity for brand-new users.

## 5. Limitations & future work
- Hit-rate is one metric on one split; add NDCG/MAP and cross-validation.
- Try matrix factorisation (SVD/ALS) and synopsis text embeddings.
- Build a Streamlit app for interactive use.
