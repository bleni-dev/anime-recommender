import streamlit as st
from recommender import df, get_recommendations

st.title("Anime Recommender")
st.caption("Content-based рекомендации по жанрам и тегам (TF-IDF + cosine similarity)")

titles = sorted(df['title'].dropna().unique())
choice = st.selectbox("Выбери аниме", titles, index=None, placeholder="Начни вводить название...")
n = st.slider("Сколько рекомендаций", 1, 20, 5)

if choice:
    found, recs = get_recommendations(choice, n)
    if found is None:
        st.warning("Не нашёл такое аниме")
    else:
        st.subheader(f"Похожие на: {found}")
        top_score = recs[0][1]
        if top_score == 0:
            st.warning("Не нашёл такое аниме")
        else:
            for rank, (title, score) in enumerate(recs, 1):
                st.write(f"{rank}. **{title}**")
                st.progress(min(score / top_score, 1.0))