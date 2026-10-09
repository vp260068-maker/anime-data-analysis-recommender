"""
Anime Recommendation System & Data Analysis
===========================================
One script that does everything:
  1. loads + cleans anime.csv and rating.csv
  2. saves the cleaned anime table  -> anime_clean.csv
  3. creates 12 visualizations      -> images/*.png
  4. builds 3 recommenders (popularity, content-based, item-based CF)
  5. (optional) evaluates CF against a popularity baseline

Usage
-----
    python anime_analysis.py                          # clean + plots + sample recommendations
    python anime_analysis.py --data-dir path/to/csvs  # where anime.csv & rating.csv live
    python anime_analysis.py --title "Death Note"     # recommendations for another title
    python anime_analysis.py --evaluate               # + offline evaluation (takes several minutes)

Dataset: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
"""
import argparse
import html
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
IMG = HERE / "images"
SEED = 42
C1, C2 = "#2a6fdb", "#e8743b"
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 200, "display.max_colwidth", 55)


# --------------------------------------------------------------------------- #
# 1. DATA LOADING & CLEANING
# --------------------------------------------------------------------------- #
def load_anime(path):
    df = pd.read_csv(path)
    df["name"] = df["name"].apply(html.unescape)          # fix &#039; etc.
    df["episodes"] = pd.to_numeric(df["episodes"].replace("Unknown", np.nan), errors="coerce")
    df["genre"] = df["genre"].fillna("")
    df["type"] = df["type"].fillna("Unknown")
    df["genre_list"] = df["genre"].apply(lambda s: [g.strip() for g in s.split(",") if g.strip()])
    return df


def load_ratings(path):
    """rating = -1 means 'watched but not scored' -> removed."""
    r = pd.read_csv(path, dtype={"user_id": "int32", "anime_id": "int32", "rating": "int8"})
    return r[r["rating"] != -1].reset_index(drop=True)


def save_clean(anime, ratings, path):
    """Cleaned anime table + simple per-anime rating statistics from rating.csv."""
    stats = ratings.groupby("anime_id")["rating"].agg(user_ratings="count", user_mean_rating="mean").round(3)
    out = anime.drop(columns="genre_list").merge(stats, on="anime_id", how="left")
    out["n_genres"] = anime["genre_list"].str.len().values
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- #
# 2. VISUALIZATIONS
# --------------------------------------------------------------------------- #
def _save(fig, name):
    IMG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=130)
    plt.close(fig)


def make_plots(anime, ratings):
    # 01 missing values
    miss = anime[["genre", "type", "episodes", "rating"]].copy()
    miss["genre"] = miss["genre"].replace("", np.nan)
    miss["type"] = miss["type"].replace("Unknown", np.nan)
    pct = miss.isna().mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(pct.index, pct.values, color=C1)
    for y, v in enumerate(pct.values):
        ax.text(v + 0.05, y, f"{v:.1f}%", va="center")
    ax.set(title="Missing values in anime.csv", xlabel="% missing")
    _save(fig, "01_missing_values.png")

    # 02 type distribution
    c = anime["type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=c.index, y=c.values, color=C1, ax=ax)
    ax.bar_label(ax.containers[0])
    ax.set(title="Number of anime by type", xlabel="", ylabel="count")
    _save(fig, "02_type_distribution.png")

    # 03 average rating distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(anime["rating"].dropna(), bins=40, kde=True, color=C1, ax=ax)
    ax.axvline(anime["rating"].mean(), color=C2, ls="--", label=f"mean = {anime['rating'].mean():.2f}")
    ax.legend()
    ax.set(title="Distribution of average anime rating", xlabel="average rating")
    _save(fig, "03_avg_rating_distribution.png")

    # 04 user rating distribution
    c = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=c.index, y=c.values, color=C1, ax=ax)
    ax.set(title="Individual user ratings (1-10, unrated excluded)", xlabel="rating", ylabel="count")
    _save(fig, "04_user_rating_distribution.png")

    # 05 top genres
    cnt = Counter(g for gl in anime["genre_list"] for g in gl).most_common(15)
    g, v = zip(*cnt)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.barplot(x=list(v), y=list(g), color=C1, ax=ax)
    ax.bar_label(ax.containers[0], padding=3)
    ax.set(title="Top 15 genres by number of anime", xlabel="count", ylabel="")
    _save(fig, "05_top_genres.png")

    # 06 highest-rated genres
    ex = anime.explode("genre_list").dropna(subset=["genre_list", "rating"])
    s = ex.groupby("genre_list")["rating"].agg(["mean", "count"])
    s = s[s["count"] >= 100].sort_values("mean", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.barplot(x=s["mean"], y=s.index, color=C2, ax=ax)
    ax.set_xlim(s["mean"].min() - 0.3, s["mean"].max() + 0.1)
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
    ax.set(title="Highest-rated genres (>= 100 anime)", xlabel="mean rating", ylabel="")
    _save(fig, "06_genre_avg_rating.png")

    # 07 rating by type
    d = anime.dropna(subset=["rating"])
    order = d.groupby("type")["rating"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=d, x="type", y="rating", order=order, color=C1, ax=ax)
    ax.set(title="Average rating by anime type", xlabel="")
    _save(fig, "07_rating_by_type.png")

    # 08 popularity vs rating
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["members"], d["rating"], s=6, alpha=.35, color=C1)
    ax.set_xscale("log")
    rho = d[["members", "rating"]].corr(method="spearman").iloc[0, 1]
    ax.set(title=f"Popularity vs rating (Spearman rho = {rho:.2f})", xlabel="members (log scale)", ylabel="average rating")
    _save(fig, "08_members_vs_rating.png")

    # 09 episodes
    e = anime["episodes"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(e[e <= 100], bins=50, color=C1, ax=ax)
    ax.set(title="Episode count (<=100 episodes)", xlabel="episodes")
    _save(fig, "09_episodes_distribution.png")

    # 10 top popular
    t = anime.nlargest(10, "members").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(t["name"].str.slice(0, 40), t["members"], color=C2)
    ax.set(title="Top 10 most popular anime (members)", xlabel="members")
    _save(fig, "10_top_popular.png")

    # 11 user / item activity (long tail)
    per_user = ratings.groupby("user_id").size()
    per_item = ratings.groupby("anime_id").size()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(per_user.clip(upper=600), bins=60, color=C1)
    ax[0].set_yscale("log")
    ax[0].set(title="Ratings per user (clipped at 600)", xlabel="# ratings", ylabel="users")
    ax[1].hist(per_item, bins=np.logspace(0, np.log10(per_item.max()), 50), color=C2)
    ax[1].set(xscale="log", yscale="log", title="Ratings per anime (long tail)", xlabel="# ratings (log)", ylabel="anime")
    _save(fig, "11_user_item_activity.png")

    # 12 genre correlation
    genres = [g for g, _ in Counter(g for gl in anime["genre_list"] for g in gl).most_common(15)]
    m = pd.DataFrame({g: anime["genre_list"].apply(lambda l: g in l) for g in genres}).astype(int)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(m.corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .7}, ax=ax)
    ax.set_title("Genre co-occurrence correlation")
    _save(fig, "12_genre_correlation.png")


# --------------------------------------------------------------------------- #
# 3. RECOMMENDERS
# --------------------------------------------------------------------------- #
class PopularityRecommender:
    """IMDB weighted rating: WR = v/(v+m)*R + m/(v+m)*C."""

    def __init__(self, anime, quantile=0.80):
        d = anime.dropna(subset=["rating"]).copy()
        C, m = d["rating"].mean(), d["members"].quantile(quantile)
        d = d[d["members"] >= m]
        d["score"] = d["members"] / (d["members"] + m) * d["rating"] + m / (d["members"] + m) * C
        self.table = d.sort_values("score", ascending=False)

    def recommend(self, n=10):
        return self.table[["anime_id", "name", "genre", "rating", "members", "score"]].head(n)


class ContentRecommender:
    """TF-IDF on genre + type, cosine similarity."""

    def __init__(self, anime):
        self.anime = anime.reset_index(drop=True)
        text = self.anime["genre"].str.replace(" ", "_").str.replace(",_", " ") + " type_" + self.anime["type"]
        self.matrix = TfidfVectorizer(token_pattern=r"[^ ]+").fit_transform(text)
        self.idx = pd.Series(self.anime.index, index=self.anime["name"]).groupby(level=0).first()

    def recommend(self, title, n=10):
        i = self.idx[title]
        sim = linear_kernel(self.matrix[i], self.matrix).ravel()
        sim[i] = -1
        order = np.lexsort((-self.anime["members"].values, -sim))[:n]   # ties -> more popular first
        out = self.anime.iloc[order][["anime_id", "name", "genre", "type", "rating"]].copy()
        out["similarity"] = sim[order]
        return out


class ItemCFRecommender:
    """Item-based collaborative filtering on mean-centred ratings (cosine)."""

    def __init__(self, ratings, anime, min_ratings=50):
        counts = ratings["anime_id"].value_counts()
        r = ratings[ratings["anime_id"].isin(counts[counts >= min_ratings].index)].copy()
        r["rating"] = r["rating"] - r.groupby("user_id")["rating"].transform("mean")
        self.users = pd.Index(r["user_id"].unique())
        self.items = pd.Index(r["anime_id"].unique())
        ui = sparse.csr_matrix(
            (r["rating"].values, (self.users.get_indexer(r["user_id"]), self.items.get_indexer(r["anime_id"]))),
            shape=(len(self.users), len(self.items)))
        self.item_vec = normalize(ui.T.tocsr())
        self.meta = anime.set_index("anime_id")

    def _frame(self, top, values, col):
        ids = self.items[top]
        return pd.DataFrame({"anime_id": ids, "name": self.meta["name"].reindex(ids).values,
                             "genre": self.meta["genre"].reindex(ids).values, col: values[top]})

    def similar(self, anime_id, n=10):
        i = self.items.get_loc(anime_id)
        sim = (self.item_vec @ self.item_vec[i].T).toarray().ravel()
        sim[i] = -1
        return self._frame(np.argsort(-sim)[:n], sim, "similarity")

    def recommend_for_liked(self, liked_ids, n=10):
        liked = [self.items.get_loc(a) for a in liked_ids if a in self.items]
        scores = np.asarray((self.item_vec @ self.item_vec[liked].T).sum(axis=1)).ravel()
        scores[liked] = -np.inf
        return self._frame(np.argsort(-scores)[:n], scores, "score")


# --------------------------------------------------------------------------- #
# 4. EVALUATION
# --------------------------------------------------------------------------- #
def hit_rate(ratings, anime, n_users=1500, k=10, like_threshold=8):
    """Hide one liked anime per unseen test user; is it recovered in the top-k?"""
    rng = np.random.default_rng(SEED)
    users = ratings["user_id"].unique()
    rng.shuffle(users)
    cut = int(len(users) * .8)
    train, test = ratings[ratings["user_id"].isin(users[:cut])], ratings[ratings["user_id"].isin(users[cut:])]
    model = ItemCFRecommender(train, anime)
    pop = train.groupby("anime_id").size().sort_values(ascending=False).index[: k + 50]
    liked = test[(test["rating"] >= like_threshold) & test["anime_id"].isin(model.items)]
    groups = liked.groupby("user_id")["anime_id"].apply(list)
    groups = groups[groups.str.len() >= 5].sample(min(n_users, len(groups)), random_state=SEED)
    h_cf = h_pop = 0
    for ids in groups:
        held = ids[rng.integers(len(ids))]
        seen = [a for a in ids if a != held]
        h_cf += held in model.recommend_for_liked(seen, k)["anime_id"].tolist()
        h_pop += held in [p for p in pop if p not in seen][:k]
    n = len(groups)
    return pd.DataFrame({"model": ["Item-based CF", "Popularity baseline"],
                         f"HitRate@{k}": [h_cf / n, h_pop / n], "users_evaluated": n})


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE), help="folder with anime.csv and rating.csv")
    ap.add_argument("--title", default="Death Note", help="anime title for sample recommendations")
    ap.add_argument("--evaluate", action="store_true", help="run offline evaluation (slow)")
    ap.add_argument("--eval-users", type=int, default=1500)
    a = ap.parse_args()
    d = Path(a.data_dir)

    print("[1/5] Loading & cleaning data ...")
    anime, ratings = load_anime(d / "anime.csv"), load_ratings(d / "rating.csv")
    print(f"      anime: {anime.shape} | ratings: {ratings.shape} | users: {ratings.user_id.nunique():,}")
    save_clean(anime, ratings, HERE / "anime_clean.csv")
    print("      saved anime_clean.csv")

    print("[2/5] Creating visualizations -> images/ ...")
    make_plots(anime, ratings)

    print("[3/5] Popularity recommender: top 10")
    print(PopularityRecommender(anime).recommend(10).to_string(index=False))

    print(f"[4/5] Content-based: similar to '{a.title}'")
    print(ContentRecommender(anime).recommend(a.title, 10).to_string(index=False))

    print(f"[5/5] Item-based CF: similar to '{a.title}'")
    cf = ItemCFRecommender(ratings, anime)
    aid = anime.loc[anime["name"] == a.title, "anime_id"].iloc[0]
    print(cf.similar(aid, 10).to_string(index=False))

    if a.evaluate:
        print("Evaluating (this takes a while) ...")
        res = hit_rate(ratings, anime, n_users=a.eval_users)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
