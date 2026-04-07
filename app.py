
import streamlit as st
import pickle
import requests
import sqlite3
import hashlib
import pandas as pd
import os
from surprise import SVD, Dataset, Reader

try:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
except:
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")

# ── Database ────────────────────────────────────
def init_db():
    conn = sqlite3.connect('cineai.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    movie_title TEXT,
                    movie_id INTEGER,
                    rating REAL,
                    FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, email, password):
    try:
        conn = sqlite3.connect('cineai.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                  (username, email, hash_password(password)))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists!"

def login(username, password):
    conn = sqlite3.connect('cineai.db')
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return (True, user) if user else (False, None)

def save_rating(user_id, movie_title, movie_id, rating):
    conn = sqlite3.connect('cineai.db')
    c = conn.cursor()
    c.execute("SELECT id FROM ratings WHERE user_id=? AND movie_title=?", (user_id, movie_title))
    if c.fetchone():
        c.execute("UPDATE ratings SET rating=? WHERE user_id=? AND movie_title=?",
                  (rating, user_id, movie_title))
    else:
        c.execute("INSERT INTO ratings (user_id, movie_title, movie_id, rating) VALUES (?, ?, ?, ?)",
                  (user_id, movie_title, movie_id, rating))
    conn.commit()
    conn.close()

def get_user_ratings(user_id):
    conn = sqlite3.connect('cineai.db')
    c = conn.cursor()
    c.execute("SELECT movie_title, rating FROM ratings WHERE user_id=?", (user_id,))
    r = c.fetchall()
    conn.close()
    return r

def get_all_ratings():
    conn = sqlite3.connect('cineai.db')
    c = conn.cursor()
    c.execute("SELECT user_id, movie_title, movie_id, rating FROM ratings")
    r = c.fetchall()
    conn.close()
    return r

# ── Retrain SVD ─────────────────────────────────
def retrain_svd():
    rows = get_all_ratings()
    if len(rows) < 5:
        return False, "Need at least 5 ratings to retrain."
    df = pd.DataFrame(rows, columns=['user_id', 'movie_title', 'movie_id', 'rating'])
    df = df.dropna(subset=['movie_id'])
    if len(df) < 5:
        return False, "Not enough valid ratings."
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(trainset)
    pickle.dump(svd, open('svd_model.pkl', 'wb'))
    return True, f"Updated on {len(df)} ratings from {df['user_id'].nunique()} users!"

# ── Load Data ───────────────────────────────────
import gdown
import os

@st.cache_resource
def load_data():
    if not os.path.exists('movies.pkl'):
        gdown.download("https://drive.google.com/uc?id=1kL19GgQ-wmr84oFZ2ASHYtbL1PEqeDam", 'movies.pkl', quiet=False)
    if not os.path.exists('similarity.pkl'):
        gdown.download("https://drive.google.com/uc?id=1X52lxt9Ugh-mnhYzb5YJ-eCwXDcQjMLc", 'similarity.pkl', quiet=False)
    if not os.path.exists('svd_model.pkl'):
        gdown.download("https://drive.google.com/uc?id=1jYMOBAS6sRjH5IO0QimfWiJN9MSZOKWC", 'svd_model.pkl', quiet=False)
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    svd_model = pickle.load(open('svd_model.pkl', 'rb'))
    return movies, similarity, svd_model

movies, similarity, svd_model = load_data()

# ── Fetch Movie Info ─────────────────────────────
def fetch_movie_info(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        data = requests.get(url).json()
        poster = data.get('Poster', '')
        return {
            'poster': poster if poster and poster != 'N/A' else 'https://via.placeholder.com/300x450?text=No+Poster',
            'rating': data.get('imdbRating', 'N/A'),
            'year': data.get('Year', ''),
        }
    except:
        return {'poster': 'https://via.placeholder.com/300x450?text=No+Poster',
                'rating': 'N/A', 'year': ''}

# ── Recommend Functions ──────────────────────────
def content_recommend(movie_title, n=10):
    try:
        idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        sim_scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
        indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[indices].tolist(), movies['movie_id'].iloc[indices].tolist()
    except:
        return [], []

def hybrid_recommend(movie_title, user_id, n=10):
    try:
        idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        sim_scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:30]
        indices = [i[0] for i in sim_scores]
        candidates = movies.iloc[indices].copy()
        candidates['content_score'] = [s[1] for s in sim_scores]
        candidates['svd_score'] = candidates['movie_id'].apply(
            lambda mid: svd_model.predict(user_id, mid).est)
        candidates['hybrid_score'] = (
            0.5 * candidates['content_score'] +
            0.5 * (candidates['svd_score'] / 5.0))
        candidates = candidates.sort_values('hybrid_score', ascending=False).head(n)
        return candidates['title'].tolist(), candidates['movie_id'].tolist()
    except:
        return content_recommend(movie_title, n)

# ── Categories & Top Movies ──────────────────────
CATEGORIES = {
    "Action": ["Avatar", "The Dark Knight Rises", "John Carter", "Spectre", "Mad Max: Fury Road"],
    "Comedy": ["The Hangover", "Superbad", "Bridesmaids", "21 Jump Street", "Knocked Up"],
    "Drama": ["The Godfather", "Forrest Gump", "The Shawshank Redemption", "Titanic", "Schindler's List"],
    "Thriller": ["Inception", "Gone Girl", "The Silence of the Lambs", "Se7en", "Shutter Island"],
    "Romance": ["The Notebook", "Pride and Prejudice", "La La Land", "Twilight", "A Walk to Remember"]
}

TOP_MOVIES = ["Avatar", "The Dark Knight Rises", "Inception", "Titanic", "The Avengers",
              "Interstellar", "Spectre", "Guardians of the Galaxy", "John Carter", "Spider-Man 3"]

# ── Page Config ──────────────────────────────────
st.set_page_config(page_title="CineVerse", page_icon="🎬", layout="wide")

# ── Session State ────────────────────────────────
for key, val in [('logged_in', False), ('user_id', None), ('username', None),
                 ('show_login', False), ('show_signup', False),
                 ('search_results', None), ('search_query', None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }
.navbar {
    background: linear-gradient(90deg, #0d0d0d, #1a1a2e);
    padding: 0 40px; display: flex; align-items: center;
    justify-content: space-between; height: 65px;
    border-bottom: 2px solid #e50914;
    box-shadow: 0 4px 20px rgba(229,9,20,0.2);
}
.nav-logo { color: #e50914; font-size: 26px; font-weight: 800; }
.nav-logo span { color: white; }
.nav-links { display: flex; gap: 30px; list-style: none; margin: 0; padding: 0; }
.nav-links a { color: #cccccc; text-decoration: none; font-size: 14px; font-weight: 500; }
.nav-links a:hover { color: #e50914; }
.hero {
    background: linear-gradient(135deg, #0d0d0d, #1a1a2e, #0f3460);
    padding: 80px 40px; text-align: center; border-bottom: 1px solid #222;
}
.hero h1 { font-size: 56px; font-weight: 800; color: white; margin: 0 0 15px 0; }
.hero h1 span { color: #e50914; }
.hero p { color: #aaaaaa; font-size: 18px; max-width: 600px; margin: 0 auto 35px; }
.hero-stats { display: flex; justify-content: center; gap: 50px; margin-top: 40px; }
.stat { text-align: center; }
.stat-number { color: #e50914; font-size: 32px; font-weight: 800; }
.stat-label { color: #888; font-size: 13px; }
.search-section { background: #111; padding: 40px; text-align: center; border-bottom: 1px solid #222; }
.search-section h2 { color: white; font-size: 24px; margin-bottom: 5px; }
.search-section p { color: #888; font-size: 14px; margin-bottom: 20px; }
.section-header { padding: 30px 0 15px 0; display: flex; align-items: center; gap: 12px; }
.section-header h2 { color: white; font-size: 22px; font-weight: 700; margin: 0; }
.section-line { flex: 1; height: 1px; background: #222; }
.section-badge { background: #e50914; color: white; padding: 3px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.movie-card { background: #1a1a1a; border-radius: 12px; overflow: hidden; border: 1px solid #2a2a2a; transition: all 0.3s; }
.movie-card:hover { transform: translateY(-5px); border-color: #e50914; box-shadow: 0 10px 30px rgba(229,9,20,0.2); }
.movie-info { padding: 10px; }
.movie-name { color: white; font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.movie-rating { color: #ffd700; font-size: 12px; font-weight: 600; }
.rank-badge { background: #e50914; color: white; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
.auth-modal { background: #161616; border: 1px solid #333; border-radius: 16px; padding: 35px; max-width: 420px; margin: 20px auto; }
.stButton > button {
    background: linear-gradient(90deg, #e50914, #c00812) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 12px 25px !important;
    font-weight: 600 !important; width: 100% !important;
}
.stSelectbox > div > div { background: #1a1a1a !important; border: 1px solid #e50914 !important; border-radius: 8px !important; }
.rec-card { background: #1a1a1a; border-radius: 10px; padding: 12px; border: 1px solid #2a2a2a; text-align: center; }
.rec-title { color: white; font-size: 12px; font-weight: 600; margin: 8px 0 4px; }
.rec-rating { color: #ffd700; font-size: 11px; }
.retrain-box { background: linear-gradient(135deg, #1a1a2e, #0f3460); border: 1px solid #e50914; border-radius: 12px; padding: 20px; margin: 20px 0; text-align: center; }
.footer { background: #0d0d0d; border-top: 2px solid #e50914; padding: 50px 40px 20px; margin-top: 60px; }
.footer-grid { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 40px; margin-bottom: 40px; }
.footer-brand h3 { color: #e50914; font-size: 24px; font-weight: 800; margin: 0 0 10px; }
.footer-brand p { color: #666; font-size: 13px; line-height: 1.7; }
.footer-col h4 { color: white; font-size: 14px; font-weight: 600; margin: 0 0 15px; }
.footer-col a { display: block; color: #666; text-decoration: none; font-size: 13px; margin-bottom: 8px; }
.footer-col a:hover { color: #e50914; }
.footer-bottom { border-top: 1px solid #1a1a1a; padding-top: 20px; text-align: center; color: #444; font-size: 12px; }
.user-badge { background: #e50914; color: white; padding: 5px 15px; border-radius: 20px; font-size: 13px; font-weight: 600; }
body { background: #0d0d0d; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ───────────────────────────────────────
logged_in = st.session_state.logged_in
username = st.session_state.username

st.markdown(f"""
<div class="navbar">
    <div class="nav-logo">🎬 Cine<span>Verse</span></div>
    <ul class="nav-links">
        <li><a href="#">Home</a></li>
        <li><a href="#">Reviews</a></li>
        <li><a href="#">Features</a></li>
        <li><a href="#">About Us</a></li>
        <li><a href="#">Contact Us</a></li>
    </ul>
    <div>{'<span class="user-badge">👤 ' + str(username) + '</span>' if logged_in else ''}</div>
</div>
""", unsafe_allow_html=True)

# ── Auth Buttons ─────────────────────────────────
if not logged_in:
    col1, col2, col3 = st.columns([6, 1, 1])
    with col2:
        if st.button("Login", key="nav_login"):
            st.session_state.show_login = True
            st.session_state.show_signup = False
    with col3:
        if st.button("Sign Up", key="nav_signup"):
            st.session_state.show_signup = True
            st.session_state.show_login = False
else:
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Logout"):
            for k in ['logged_in', 'user_id', 'username']:
                st.session_state[k] = False if k == 'logged_in' else None
            st.session_state.search_results = None
            st.rerun()

# ── Login Modal ──────────────────────────────────
if st.session_state.show_login and not logged_in:
    st.markdown('<div class="auth-modal">', unsafe_allow_html=True)
    st.markdown("### 🔐 Welcome Back")
    u = st.text_input("Username", key="li_user")
    p = st.text_input("Password", type="password", key="li_pass")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login", key="do_login"):
            ok, user = login(u, p)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]
                st.session_state.show_login = False
                st.rerun()
            else:
                st.error("Invalid credentials!")
    with c2:
        if st.button("Cancel", key="cancel_login"):
            st.session_state.show_login = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Signup Modal ─────────────────────────────────
if st.session_state.show_signup and not logged_in:
    st.markdown('<div class="auth-modal">', unsafe_allow_html=True)
    st.markdown("### 📝 Create Account")
    nu = st.text_input("Username", key="su_user")
    ne = st.text_input("Email", key="su_email")
    np_ = st.text_input("Password", type="password", key="su_pass")
    nc = st.text_input("Confirm Password", type="password", key="su_confirm")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create Account", key="do_signup"):
            if np_ != nc:
                st.error("Passwords don't match!")
            elif len(np_) < 6:
                st.error("Password too short!")
            else:
                ok, msg = signup(nu, ne, np_)
                if ok:
                    st.success(msg)
                    st.session_state.show_signup = False
                    st.session_state.show_login = True
                    st.rerun()
                else:
                    st.error(msg)
    with c2:
        if st.button("Cancel", key="cancel_signup"):
            st.session_state.show_signup = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Discover Movies<br>You'll <span>Love</span></h1>
    <p>Browse thousands of movies, get personalized picks, and share your reviews with the world.</p>
    <div class="hero-stats">
        <div class="stat"><div class="stat-number">4,800+</div><div class="stat-label">Movies</div></div>
        <div class="stat"><div class="stat-number">50K+</div><div class="stat-label">Reviews</div></div>
        <div class="stat"><div class="stat-number">10K+</div><div class="stat-label">Users</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Search ───────────────────────────────────────
st.markdown("""
<div class="search-section">
    <h2>🔍 Find Your Next Favourite Movie</h2>
    <p>Enter the movie you want to search and get recommendations</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    movie_list = sorted(movies['title'].dropna().tolist())
    selected = st.selectbox("", movie_list, label_visibility="hidden", key="main_search")
    if st.button("🔍 Search Movie", key="search_btn"):
        with st.spinner("Finding movies for you..."):
            if st.session_state.logged_in:
                titles, movie_ids = hybrid_recommend(selected, st.session_state.user_id)
            else:
                titles, movie_ids = content_recommend(selected)
            # ✅ Save results in session state so rating doesn't clear them
            st.session_state.search_results = list(zip(titles, movie_ids))
            st.session_state.search_query = selected

# ✅ Show results from session state — stays even after rating!
if st.session_state.search_results:
    results = st.session_state.search_results
    titles = [r[0] for r in results]
    movie_ids = [r[1] for r in results]

    st.markdown(f"""
    <div class="section-header">
        <h2>Results for "{st.session_state.search_query}"</h2>
        <div class="section-line"></div>
        <span class="section-badge">{'Personalised for you' if logged_in else 'Recommended'}</span>
    </div>
    """, unsafe_allow_html=True)

    for row_start in [0, 5]:
        cols = st.columns(5)
        for i in range(5):
            idx = row_start + i
            if idx < len(titles):
                info = fetch_movie_info(titles[idx])
                with cols[i]:
                    st.markdown(f"""
                    <div class="rec-card">
                        <img src="{info['poster']}" style="width:100%;border-radius:8px;">
                        <div class="rec-title">{titles[idx]}</div>
                        <div class="rec-rating">⭐ {info['rating']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if logged_in:
                        r = st.slider("Rate", 1.0, 5.0, 3.0, 0.5, key=f"rs_{idx}")
                        if st.button("⭐ Save Rating", key=f"sv_{idx}"):
                            save_rating(st.session_state.user_id,
                                      titles[idx], movie_ids[idx], r)
                            st.success(f"✅ Rated '{titles[idx]}'!")
                    else:
                        st.caption("🔐 Login to rate")

st.markdown("<br>", unsafe_allow_html=True)

# ── Top Movies ───────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2>🏆 Top Movies Right Now</h2>
    <div class="section-line"></div>
    <span class="section-badge">Top 10</span>
</div>
""", unsafe_allow_html=True)

for row_start, rank_offset in [(0, 1), (5, 6)]:
    cols = st.columns(5)
    for i in range(5):
        title = TOP_MOVIES[row_start + i]
        info = fetch_movie_info(title)
        with cols[i]:
            st.markdown(f"""
            <div class="movie-card">
                <div style="position:relative;">
                    <img src="{info['poster']}" style="width:100%;border-radius:10px 10px 0 0;">
                    <div style="position:absolute;top:8px;left:8px;">
                        <span class="rank-badge">#{rank_offset + i}</span>
                    </div>
                </div>
                <div class="movie-info">
                    <div class="movie-name">{title}</div>
                    <div class="movie-rating">⭐ {info['rating']}</div>
                    <div style="color:#666;font-size:11px;">{info['year']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Categories ───────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2>🎭 Browse by Category</h2>
    <div class="section-line"></div>
    <span class="section-badge">Explore</span>
</div>
""", unsafe_allow_html=True)

selected_cat = st.radio("", list(CATEGORIES.keys()), horizontal=True, label_visibility="hidden")
cols = st.columns(5)
for i, title in enumerate(CATEGORIES[selected_cat]):
    info = fetch_movie_info(title)
    with cols[i]:
        st.markdown(f"""
        <div class="movie-card">
            <img src="{info['poster']}" style="width:100%;border-radius:10px 10px 0 0;">
            <div class="movie-info">
                <div class="movie-name">{title}</div>
                <div class="movie-rating">⭐ {info['rating']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── My Ratings ───────────────────────────────────
if logged_in:
    st.markdown("""
    <div class="section-header">
        <h2>❤️ My Rated Movies</h2>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    ratings = get_user_ratings(st.session_state.user_id)
    if ratings:
        r_cols = st.columns(3)
        for i, (title, rating) in enumerate(ratings):
            with r_cols[i % 3]:
                stars = "⭐" * int(rating)
                st.markdown(f"""
                <div style="background:#1a1a1a;padding:15px;border-radius:10px;
                            margin:5px;border-left:3px solid #e50914;">
                    <b style="color:white;">🎬 {title}</b><br>
                    <span style="color:#ffd700;">{stars} {rating}/5</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="retrain-box">
            <h3 style="color:white;margin:0 0 8px;">🔄 Update My Recommendations</h3>
            <p style="color:#aaa;font-size:13px;margin:0 0 15px;">
                Click below to personalise recommendations based on your latest ratings.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Update Recommendations", key="retrain_btn"):
            with st.spinner("Updating..."):
                ok, msg = retrain_svd()
            if ok:
                st.success(f"✅ {msg}")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.warning(f"⚠️ {msg}")
    else:
        st.info("Search for movies, rate them and they'll appear here!")

# ── Footer ───────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-grid">
        <div class="footer-brand">
            <h3>🎬 CineVerse</h3>
            <p>Your go-to destination for movie discovery. We help you find films you'll love,
            read honest reviews, and connect with a community of movie enthusiasts.</p>
            <div style="display:flex;gap:12px;margin-top:15px;">
                <a style="color:#888;text-decoration:none;font-size:20px;" href="#">📘</a>
                <a style="color:#888;text-decoration:none;font-size:20px;" href="#">🐦</a>
                <a style="color:#888;text-decoration:none;font-size:20px;" href="#">📸</a>
                <a style="color:#888;text-decoration:none;font-size:20px;" href="#">▶️</a>
            </div>
        </div>
        <div class="footer-col">
            <h4>Quick Links</h4>
            <a href="#">Home</a>
            <a href="#">Browse Movies</a>
            <a href="#">Top Rated</a>
            <a href="#">New Releases</a>
            <a href="#">Reviews</a>
        </div>
        <div class="footer-col">
            <h4>Company</h4>
            <a href="#">About Us</a>
            <a href="#">Careers</a>
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Use</a>
        </div>
        <div class="footer-col">
            <h4>Contact Us</h4>
            <a href="#">📧 hello@cineverse.com</a>
            <a href="#">📞 +91 98765 43210</a>
            <a href="#">📍 Mumbai, India</a>
            <a href="#">💬 Live Chat</a>
        </div>
    </div>
    <div class="footer-bottom">
        <p>© 2026 CineVerse. All rights reserved. Made with ❤️ for movie lovers.</p>
    </div>
</div>
""", unsafe_allow_html=True)
