
import html

import pandas as pd
import streamlit as st

from recommender import get_recommendations, meta

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎴",
    layout="wide"
)

# ----------------------------------------------------------------------------
# DESIGN
# ----------------------------------------------------------------------------

CSS = """
:root {
    --accent: #65aef5;
    --paper: #e8edf5;
    --muted: #929baa;
    --card: #191e27;
    --line: #2b3340;
}

html, body, .stApp {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #11151c;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.eyebrow {
    color: var(--accent);
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .12em;
}

.hero {
    color: var(--paper);
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.15;
    margin: .5rem 0;
}

.hero span {
    color: var(--accent);
}

.lead {
    color: var(--muted);
    max-width: 600px;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

.section {
    color: var(--paper);
    font-size: 1.45rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem;
}

.hint {
    color: var(--muted);
    font-size: .82rem;
    margin-top: .7rem;
}

/* Selected anime */
.now {
    display: flex;
    gap: 16px;
    align-items: center;
    padding: 14px;
    margin: 1rem 0;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
}

.now img {
    width: 85px;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border-radius: 6px;
    flex-shrink: 0;
}

.now .t {
    color: var(--paper);
    font-weight: 600;
    font-size: 1.1rem;
    margin: 4px 0;
}

.label {
    color: var(--accent);
    font-size: .7rem;
    font-weight: 600;
}

.chip {
    display: inline-block;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: .72rem;
    color: var(--muted);
    margin: 6px 5px 0 0;
}

/* Recommendation cards */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
    gap: 14px;
}

.card {
    display: block;
    text-decoration: none !important;
    color: var(--paper) !important;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color .15s;
}

.card:hover {
    border-color: var(--accent);
}

.poster {
    position: relative;
    aspect-ratio: 2 / 3;
    background: #222a35;
}

.poster img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.noimg {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: .8rem;
}

.rank {
    position: absolute;
    top: 8px;
    left: 8px;
    background: #202b39;
    color: var(--accent);
    font-weight: 600;
    font-size: .72rem;
    padding: 3px 8px;
    border-radius: 5px;
}

.info {
    padding: 10px;
}

.name {
    color: var(--paper);
    font-weight: 600;
    font-size: .86rem;
    line-height: 1.35;
    margin-bottom: 5px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.sub {
    color: var(--muted);
    font-size: .75rem;
}

.bar {
    height: 3px;
    background: var(--line);
    margin-top: 9px;
    border-radius: 2px;
}

.bar > span {
    display: block;
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
}

.foot {
    color: var(--muted);
    font-size: .78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
}

.foot a {
    color: var(--accent) !important;
}

/* Streamlit inputs */
.stSelectbox label,
.stSlider label {
    color: var(--paper) !important;
}

div[data-baseweb="select"] {
    background: var(--card);
    border-radius: 8px;
}

div[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 8px;
}
"""

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------

@st.cache_resource
def load_lookup():
    return meta.drop_duplicates("title").set_index("title").to_dict("index")


@st.cache_resource
def load_titles():
    return meta.sort_values(
        "popularity", ascending=False
    )["title"].tolist()


lookup = load_lookup()
titles = load_titles()


def esc(text):
    return html.escape(str(text)).replace("$", "&#36;")


def has(value):
    if isinstance(value, str):
        return value.strip() != ""
    return pd.notna(value)


def year_score(info):
    parts = []

    if has(info.get("year")):
        parts.append(str(int(info["year"])))

    if has(info.get("score")):
        parts.append(f"★ {int(info['score'])}")

    return " · ".join(parts)


def poster(info):
    cover = info.get("cover")

    if isinstance(cover, str) and cover:
        return f'<img src="{esc(cover)}" loading="lazy" alt="">'

    return '<div class="noimg">No cover</div>'


def card(rank, title, score, top):
    info = lookup.get(title, {})
    pct = max(0, min(100, score / top * 100)) if top else 0

    english = info.get("title_english")
    tip = esc(english) if has(english) else esc(title)

    link = esc(info["url"]) if has(info.get("url")) else "#"

    return (
        f'<a class="card" href="{link}" target="_blank" title="{tip}">'
        f'<div class="poster">{poster(info)}'
        f'<span class="rank">{rank:02d}</span></div>'
        f'<div class="info"><div class="name">{esc(title)}</div>'
        f'<div class="sub">{esc(year_score(info))}</div>'
        f'<div class="bar"><span style="width:{pct:.0f}%"></span></div>'
        f'</div></a>'
    )


# ----------------------------------------------------------------------------
# PAGE
# ----------------------------------------------------------------------------

st.markdown(
    '<div class="eyebrow">ANIME RECOMMENDER</div>'
    '<div class="hero">Find your next <span>anime</span></div>'
    '<div class="lead">Pick a title you love and discover '
    'similar anime. Recommendations combine tags, synopsis text '
    'and AniList community recommendations.</div>',
    unsafe_allow_html=True,
)

col_search, col_n = st.columns([3, 1])

with col_search:
    choice = st.selectbox(
        "Pick an anime",
        titles,
        index=None,
        placeholder="Start typing a title..."
    )

with col_n:
    n = st.slider("How many", 5, 20, 10)


if choice:
    found, recs = get_recommendations(choice, n)

    if found is None:
        st.warning("Couldn't find that title.")

    else:
        info = lookup.get(found, {})

        genres = info.get("genres")
        chips = ""

        if isinstance(genres, str):
            chips = "".join(
                f'<span class="chip">{esc(g)}</span>'
                for g in genres.split(";")[:4]
            )

        st.markdown(
            f'<div class="now">{poster(info)}<div>'
            f'<div class="label">SELECTED ANIME</div>'
            f'<div class="t">{esc(found)}</div>'
            f'<div class="sub">{esc(year_score(info))}</div>'
            f'{chips}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section">Recommended for you</div>',
            unsafe_allow_html=True
        )

        top = recs[0][1] if recs else 1

        cards = "".join(
            card(i, t, s, top)
            for i, (t, s) in enumerate(recs, 1)
        )

        st.markdown(
            f'<div class="grid">{cards}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="hint">The bar shows relative similarity '
            'to the highest-ranked recommendation.</div>',
            unsafe_allow_html=True
        )

else:
    st.markdown(
        '<div class="hint">Choose an anime to get started. '
        'Popular titles appear first.</div>',
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------------
# HOW IT WORKS
# ----------------------------------------------------------------------------

with st.expander("How it works"):
    st.markdown(
        """
**Content signal.** Each anime is represented by weighted AniList tags
and genres (spoiler tags removed), together with TF-IDF of its synopsis.
Similarity is calculated using cosine similarity.

**Fan signal.** AniList users recommend titles to each other.
These links form a graph. Direct recommendations and indirect connections
add a bonus to content similarity.

**Evaluation.** With 20% of fan links hidden, the share of titles whose
top-10 contained at least one hidden link rose from 0.29 (content only)
to 0.56 (hybrid).
"""
    )


# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------

st.markdown(
    '<div class="foot">Data: '
    '<a href="https://anilist.co" target="_blank">AniList</a> · '
    'Code: '
    '<a href="https://github.com/bleni-dev/anime-recommender" '
    'target="_blank">GitHub</a></div>',
    unsafe_allow_html=True,
)
