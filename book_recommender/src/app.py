import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader
from preprocessing import load_data, prepare_data_for_nn
from content_based import ContentBasedRecommender
from neural_network import RecommenderNet, BookDataset, train_model
import numpy as np
import random

# Page Config
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&display=swap');

    /* Global Styles */
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.15) 0px, transparent 50%);
        background-attachment: fixed;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    
    p, div, span, label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }

    /* Hero Section */
    .hero-container {
        padding: 6rem 2rem;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 32px;
        margin-bottom: 4rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-weight: 800;
        font-size: 4.5rem;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        line-height: 1.1;
        text-shadow: 0 0 40px rgba(255, 255, 255, 0.1);
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.8;
        font-weight: 300;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .sidebar-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.5rem;
    }

    /* Button Styling */
    /* Button Styling */
    .stButton > button {
        background: #334155 !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        background: #475569 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        color: #ffffff !important;
        border-color: #64748b !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    .stButton > button:disabled {
        background: #1e293b !important;
        color: #64748b !important;
        border-color: #334155 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
        opacity: 0.7 !important;
    }

    /* Card Styling */
    .book-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 1.75rem;
        height: 100%;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem; /* Add spacing between vertically stacked cards */
    }
    
    .book-card:hover {
        transform: translateY(-8px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        background: rgba(30, 41, 59, 0.6);
    }
    
    .book-card::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(800px circle at var(--mouse-x) var(--mouse-y), rgba(255, 255, 255, 0.06), transparent 40%);
        opacity: 0;
        transition: opacity 0.5s;
        pointer-events: none; /* Fix: Allow clicks to pass through the hover effect */
    }
    
    .book-card:hover::after {
        opacity: 1;
    }

    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f8fafc;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }

    .book-genre {
        font-size: 0.75rem;
        color: #ec4899;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .book-desc {
        font-size: 0.925rem;
        color: #94a3b8;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }

    .book-rating {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-top: 1.25rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: auto;
    }
    
    .rating-badge {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-weight: 700;
        font-size: 0.9rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.35rem 0.75rem;
        border-radius: 12px;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: rgba(30, 41, 59, 0.5);
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 0.25rem;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #6366f1;
    }

    /* Dropdown Menu Styling - Aggressive Overrides */
    div[data-baseweb="popover"] {
        background-color: #0f1116 !important;
        border: 1px solid #334155 !important;
    }

    div[data-baseweb="popover"] > div {
        background-color: #0f1116 !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #0f1116 !important;
    }

    li[role="option"] {
        background-color: #0f1116 !important;
        color: #e2e8f0 !important;
    }

    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #334155 !important;
        color: #ffffff !important;
    }
    
    /* Target the text content span directly if needed */
    li[role="option"] span {
        color: #e2e8f0 !important;
    }
    
    /* Remove any default white backgrounds */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .section-desc {
        color: #94a3b8;
        margin-bottom: 3rem;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Input Field Styling */
    div[data-baseweb="input"] {
        background-color: #0f1116 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    input {
        color: #f8fafc !important;
        caret-color: #6366f1 !important;
    }
    
    /* Remove default white background from input container */
    div[data-baseweb="base-input"] {
        background-color: transparent !important;
    }
    
    /* Login Form Styling */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    summary {
        list-style: none;
        padding: 0.5rem;
    }
    
    summary::-webkit-details-marker {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_and_prep_data():
    # Load real data from CSVs
    books, ratings = load_data()
    ratings, num_users, num_books, user2user_encoded, book2book_encoded = prepare_data_for_nn(ratings)
    return books, ratings, num_users, num_books, user2user_encoded, book2book_encoded

@st.cache_resource
def train_nn_model(ratings, num_users, num_books):
    dataset = BookDataset(
        ratings['user_encoded'].values,
        ratings['book_encoded'].values,
        ratings['rating'].values
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = RecommenderNet(num_users, num_books)
    train_model(model, train_loader, epochs=5)
    return model

@st.cache_resource
def get_content_model_v4(books):
    return ContentBasedRecommender(books)

def render_book_card(title, author, genres, description, image_url=None, rating=None, is_prediction=False, explanation=None):
    rating_html = ""
    if rating is not None:
        rating_val = float(rating)
        color = "#10b981" if rating_val >= 4.0 else "#f59e0b" if rating_val >= 3.0 else "#ef4444"
        rating_html = f"""<div class="book-rating"><span style="color: #94a3b8; font-size: 0.75rem;">{'Predicted' if is_prediction else 'Rating'}</span><div class="rating-badge" style="color: {color}"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>{rating_val:.1f}</div></div>"""
    elif explanation:
         rating_html = f"""<div class="book-rating"><span style="color: #94a3b8; font-size: 0.75rem;">Why this book?</span><div class="rating-badge" style="color: #6366f1; font-size: 0.75rem;">{explanation}</div></div>"""
    else:
        # Placeholder for alignment
        rating_html = f"""<div class="book-rating"><span style="color: #94a3b8; font-size: 0.75rem;">Match</span><div class="rating-badge" style="color: #6366f1">Recommended</div></div>"""

    image_html = ""
    if image_url and str(image_url) != 'nan':
        image_html = f"""<div style="height: 200px; overflow: hidden; border-radius: 8px; margin-bottom: 1rem; display: flex; justify-content: center; align-items: center; background: #0f1116;">
            <img src="{image_url}" style="height: 100%; width: auto; object-fit: cover;" alt="{title}">
        </div>"""

    genre_html = ""
    if genres and str(genres) != 'nan':
        try:
            clean_genres = str(genres).replace("[", "").replace("]", "").replace("'", "").split(", ")
            top_genres = clean_genres[:3]
            genre_spans = "".join([f'<span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-right: 6px; display: inline-block; margin-bottom: 4px;">{g}</span>' for g in top_genres])
            genre_html = f'<div style="margin-bottom: 0.75rem;">{genre_spans}</div>'
        except:
            pass

    return f"""
    <div class="book-card">
        {image_html}
        <div>
            <div class="book-genre" style="background: transparent; color: #ec4899; padding: 0; margin-bottom: 0.25rem; font-size: 0.8rem; font-weight: 600;">{author}</div>
            <div class="book-title">{title}</div>
            {genre_html}
            <details style="margin-bottom: 1rem; border: 1px solid #334155; border-radius: 8px; padding: 0.5rem; background: #0f1116;">
                <summary style="cursor: pointer; color: #94a3b8; font-size: 0.875rem; font-weight: 500;">📖 View Description</summary>
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #cbd5e1; line-height: 1.6;">{description}</div>
            </details>
        </div>
        {rating_html}
    </div>
    """

def get_user_favorite_genre(user_id, ratings, books):
    user_ratings = ratings[ratings['user_id'] == user_id]
    if user_ratings.empty:
        return None
    
    # Merge with books to get genres
    user_books = user_ratings.merge(books, on='book_id')
    # Count genres for highly rated books (>3)
    high_rated = user_books[user_books['rating'] >= 4.0]
    if high_rated.empty:
        high_rated = user_books # Fallback to all books
        
    if high_rated.empty:
        return None
        
    top_genre = high_rated['author'].mode().iloc[0]
    return top_genre

def login_page(valid_user_ids):
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; margin-bottom: 2rem;'>
            <h1 style='font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, #fff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Welcome Back</h1>
            <p style='color: #94a3b8; font-size: 1.2rem;'>Sign in to access your personalized library</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            st.markdown('<label style="color: #e2e8f0; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; display: block;">User ID</label>', unsafe_allow_html=True)
            user_id_input = st.text_input("User ID", label_visibility="collapsed", placeholder="Enter your User ID (e.g., 84)")
            st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
            submit = st.form_submit_button("Sign In")
            
            if submit:
                try:
                    user_id = int(user_id_input)
                    if user_id in valid_user_ids:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.rerun()
                    else:
                        st.error("Invalid User ID. Please try again.")
                except ValueError:
                    st.error("Please enter a valid numeric User ID.")
        
        st.markdown("""
            <div style='text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
                <p style='font-size: 0.8rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>Available Demo Accounts</p>
                <div style='display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap;'>
                    <span style='background: rgba(99, 102, 241, 0.1); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.2);'>ID: 1</span>
                    <span style='background: rgba(99, 102, 241, 0.1); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.2);'>ID: 128</span>
                    <span style='background: rgba(99, 102, 241, 0.1); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.2);'>ID: 450</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        
    # Load Data (Global)
    with st.spinner("Loading System..."):
        books, ratings, num_users, num_books, user2user_encoded, book2book_encoded = load_and_prep_data()
    
    if not st.session_state.logged_in:
        login_page(ratings['user_id'].unique())
    else:
        # Dashboard Logic
        
        # Train/Load Models
        with st.spinner("Fine-tuning Neural Network..."):
            nn_model = train_nn_model(ratings, num_users, num_books)
            
        with st.spinner("Analyzing Content Features..."):
            content_model = get_content_model_v4(books)

        # Hero Section
        st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Discover Your Next Great Read</div>
            <div class="hero-subtitle">Our AI-powered engine analyzes your reading patterns to curate a personalized library just for you.</div>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.markdown(f'<div class="sidebar-header">User Profile</div>', unsafe_allow_html=True)
            
            # User Info Card
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 2rem;">
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">Logged in as</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">User #{st.session_state.user_id}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Sign Out"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.rerun()
                
            st.markdown('<div class="sidebar-header">Statistics</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Books", len(books))
            with c2:
                st.metric("Users", num_users)
            st.metric("Total Ratings", len(ratings))
            
            st.markdown("---")
            st.markdown('<div style="font-size: 0.75rem; color: #64748b; text-align: center;">Powered by PyTorch & Streamlit</div>', unsafe_allow_html=True)

        # Tabs for Navigation
        tab_trending, tab1, tab2, tab3 = st.tabs(["🔥 Trending Now", "🧠 Neural Picks", "🔍 Content Matches", "📝 Search by Description"])

        user_id = st.session_state.user_id

        # Tab 0: Trending Now
        with tab_trending:
            st.markdown('<div class="section-header"><span>🔥</span> Trending Now</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-desc">The top 10 most popular books in our community right now.</div>', unsafe_allow_html=True)
            
            # Get top 10 books by number of ratings
            top_book_ids = ratings['book_id'].value_counts().head(10).index
            
            # Create a 3-column grid for a more spacious, premium look
            cols = st.columns(3, gap="large")
            for i, book_id in enumerate(top_book_ids):
                book_info = books[books['book_id'] == book_id].iloc[0]
                col_idx = i % 3
                
                # Use the book's average rating
                avg_rating = book_info['average_rating']
                
                with cols[col_idx]:
                    st.markdown(render_book_card(
                        book_info['title'],
                        book_info['author'],
                        book_info.get('genres'),
                        book_info['description'],
                        book_info.get('image_url'),
                        rating=avg_rating,
                        explanation=f"#{i+1} Popular"
                    ), unsafe_allow_html=True)

        # Tab 1: Neural Network Recommendations
        with tab1:
            st.markdown('<div class="section-header"><span>🧠</span> Neural Engine Picks</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Deep learning predictions based on your unique reading history and similar users.</div>', unsafe_allow_html=True)
            
            user_encoded = user2user_encoded.get(user_id)
            if user_encoded is not None:
                all_book_ids = torch.tensor(list(book2book_encoded.values()), dtype=torch.long)
                user_tensor = torch.tensor([user_encoded] * len(all_book_ids), dtype=torch.long)
                
                nn_model.eval()
                with torch.no_grad():
                    predictions = nn_model(user_tensor, all_book_ids).squeeze()
                
                # Get top 3 for grid
                top_indices = predictions.argsort(descending=True)[:3].numpy()
                encoded2book = {v: k for k, v in book2book_encoded.items()}
                
                # Determine explanation
                fav_genre = get_user_favorite_genre(user_id, ratings, books)
                explanation_text = f"Matches your interest in {fav_genre}" if fav_genre else "Trending among similar users"
                
                cols = st.columns(3)
                for i, idx in enumerate(top_indices):
                    book_id = encoded2book[idx]
                    book_info = books[books['book_id'] == book_id].iloc[0]
                    with cols[i]:
                        st.markdown(render_book_card(
                            book_info['title'],
                            book_info['author'],
                            book_info.get('genres'),
                            book_info['description'],
                            book_info.get('image_url'),
                            predictions[idx],
                            is_prediction=True,
                            explanation=explanation_text
                        ), unsafe_allow_html=True)
            else:
                st.warning("New user? No history found.")

        # Tab 2: Content Based Recommendations
        with tab2:
            st.markdown('<div class="section-header"><span>🔍</span> Content Matches</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Books with similar themes and narratives to your favorites.</div>', unsafe_allow_html=True)
            
            user_ratings = ratings[ratings['user_id'] == user_id]
            if not user_ratings.empty:
                top_book_id = user_ratings.sort_values('rating', ascending=False).iloc[0]['book_id']
                top_book_title = books[books['book_id'] == top_book_id]['title'].values[0]
                
                st.info(f"Because you enjoyed **{top_book_title}**")
                
                recs_with_scores = content_model.get_recommendations(top_book_title)
                # Limit to 3 for grid
                recs_with_scores = recs_with_scores[:3]
                
                cols = st.columns(3)
                for i, (rec_title, score) in enumerate(recs_with_scores):
                    book_info = books[books['title'] == rec_title].iloc[0]
                    match_percentage = f"{int(score * 100)}% Match"
                    with cols[i]:
                        st.markdown(render_book_card(
                            book_info['title'],
                            book_info['author'],
                            book_info.get('genres'),
                            book_info['description'],
                            book_info.get('image_url'),
                            explanation=match_percentage
                        ), unsafe_allow_html=True)
            else:
                st.write("Rate some books to get content-based recommendations!")

        # Tab 3: Search by Description
        with tab3:
            st.markdown('<div class="section-header"><span>📝</span> Find by Description</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Describe what you want to read, and we\'ll find the closest matches.</div>', unsafe_allow_html=True)
            
            search_query = st.text_area("Enter a description (e.g., 'A mystery about a detective in London')", height=100)
            
            if st.button("Search"):
                if search_query:
                    with st.spinner("Searching books..."):
                        # Clear cache if method is missing (hacky fix for dev)
                        if not hasattr(content_model, 'recommend_by_description'):
                            st.cache_resource.clear()
                            content_model = get_content_model_v4(books)
                        
                        desc_recs = content_model.recommend_by_description(search_query, top_n=3)
                        
                    if desc_recs:
                        cols = st.columns(3)
                        for i, (rec_title, score) in enumerate(desc_recs):
                            book_info = books[books['title'] == rec_title].iloc[0]
                            match_percentage = f"{int(score * 100)}% Match"
                            with cols[i]:
                                st.markdown(render_book_card(
                                    book_info['title'],
                                    book_info['author'],
                                    book_info.get('genres'),
                                    book_info['description'],
                                    book_info.get('image_url'),
                                    explanation=match_percentage
                                ), unsafe_allow_html=True)
                    else:
                        st.info("No matches found. Try a different description.")
                else:
                    st.warning("Please enter a description first.")

if __name__ == "__main__":
    main()
