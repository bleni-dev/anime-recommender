
import html
import pandas as pd
import streamlit as st
from recommender import get_recommendations, meta

st.set_page_config(page_title="Anime Recommender", page_icon="🎴", layout="wide")

CSS = """
<style>
:root {--blue:#55bfff;--bg:#17191f;--card:#20232b;--text:#eee;--muted:#999;}
html,body,.stApp {font-family:Arial,sans-serif;}
#MainMenu,footer,header {visibility:hidden;}
.block-container {max-width:1100px;padding-top:2rem;}
.hero {font-size:3rem;font-weight:700;margin-bottom:8px;}
.hero span {color:var(--blue);}
.lead,.hint,.sub {color:var(--muted);}
.section {font-size:1.5rem;font-weight:600;margin:25px 0 12px;}
.now {display:flex;gap:16px;align-items:center;padding:14px;background:var(--card);border-radius:10px;margin:20px 0;}
.now img {width:85px;border-radius:6px;}
.t {font-size:1.1rem;font-weight:600;margin:5px 0;}
.chip {display:inline-block;background:#303641;color:#bbb;padding:4px 9px;border-radius:12px;margin:5px 4px 0 0;font-size:12px;}
.grid {display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:15px;}
.card {background:var(--card);border-radius:9px;overflow:hidden;color:var(--text)!important;text-decoration:none!important;}
.card:hover {outline:1px solid var(--blue);}
.poster {aspect-ratio:2/3;position:relative;background:#252a33;}
.poster img {width:100%;height:100%;object-fit:cover;}
.noimg {display:grid;place-items:center;height:100%;color:var(--muted);}
.rank {position:absolute;top:7px;left:7px;background:var(--blue);color:#111;padding:3px 7px;border-radius:4px;font-size:12px;}
.info {padding:10px;}
.name {font-weight:600;font-size:14px;min-height:35px;}
.sub {font-size:12px;margin-top:5px;}
.bar {height:3px;background:#353943;margin-top:10px;}
.bar span {display:block;height:100%;background:var(--blue);}
.foot {color:var(--muted);margin-top:35px;font-size:12px;}
.foot a {color:var(--blue);}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    lookup = meta.drop_duplicates("title").set_index("title").to_dict("index")
    titles = meta.sort_values("popularity", ascending=False)["title"].tolist()
    return lookup, titles

lookup, titles = load_data()

def esc(value):
    return html.escape(str(value)).replace("$", "&#36;")

def has(value):
    if isinstance(value, str):
        return value.strip() != ""
    return pd.notna(value)

def get_recommendations(title, n=10):
    i = find_index(title)
    if i is None:
        return None, []
    n = min(n, neighbors.shape[1])
    result = [
        (meta.loc[j, 'title'], float(s))
        for j, s in zip(neighbors[i, :n], scores[i, :n])
        if j >= 0
    ]
    return meta.loc[i, 'title'], result

def poster(info):
    cover = info.get("cover")
    return f'<img src="{esc(cover)}" loading="lazy">' if has(cover) else '<div class="noimg">No cover</div>'

def card(rank, title, score, top):
    info = lookup.get(title, {})
    pct = max(0, min(100, score / top * 100)) if top else 0
    link = esc(info.get("url", "#"))
    return (
        f'<a class="card" href="{link}" target="_blank">'
        f'<div class="poster">{poster(info)}<span class="rank">{rank}</span></div>'
        f'<div class="info"><div class="name">{esc(title)}</div>'
        f'<div class="sub">{esc(get_anime_details(info))}</div>'
        f'<div class="bar"><span style="width:{pct:.0f}%"></span></div></div></a>'
    )

st.markdown(
    '<div class="hero">Find your next <span>anime</span></div>'
    '<div class="lead">Choose an anime you like and discover similar titles.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 1])
with col1:
    choice = st.selectbox("Choose anime", titles, index=None, placeholder="Search anime...")
with col2:
    n = st.slider("Recommendations", 5, 20, 10)

if choice:
    found, recs = get_recommendations(choice, n)

    if found is None:
        st.warning("Anime not found.")
    else:
        info = lookup.get(found, {})
        genres = info.get("genres", "")
        chips = "".join(
            f'<span class="chip">{esc(g)}</span>'
            for g in genres.split(";")[:4]
        ) if isinstance(genres, str) else ""

        st.markdown(
            f'<div class="now">{poster(info)}<div>'
            f'<div class="sub">Selected anime</div>'
            f'<div class="t">{esc(found)}</div>'
            f'<div class="sub">{esc(get_anime_details(info))}</div>{chips}</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section">You might also like</div>', unsafe_allow_html=True)
        top = recs[0][1] if recs else 1
        cards = "".join(card(i, t, s, top) for i, (t, s) in enumerate(recs, 1))
        st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)
else:
    st.caption("Start by searching for your favorite anime.")

with st.expander("How it works"):
    st.markdown(
        """
**Content signal.** Each anime is a vector built from weighted AniList tags and genres
(spoiler tags removed) plus TF-IDF of the synopsis. Similarity is the cosine between vectors.

**Fan signal.** AniList users recommend titles to each other. These links form a graph:
direct recommendations and "friend of a friend" links add a bonus on top of content similarity.

**Evaluation.** With 20% of fan links hidden, the share of titles whose top-10 contained
at least one hidden link rose from 0.29 (content only) to 0.56 (hybrid).
"""
    )

st.markdown(
    '<div class="foot">Data: <a href="https://anilist.co">AniList</a> · '
    'Code: <a href="https://github.com/bleni-dev/anime-recommender">GitHub</a></div>',
    unsafe_allow_html=True
)
