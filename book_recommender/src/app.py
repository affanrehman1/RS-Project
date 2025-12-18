import streamlit as st
import os
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from preprocessing import load_data, prepare_data_for_nn
from content_based import ContentBasedRecommender
from neural_network import RecommenderNet, BookDataset, train_model
from database import (
    create_user,
    add_rating,
    get_valid_user_ids,
    get_user_ratings,
    get_system_stats,
    get_book_average_rating,
)
import numpy as np
import random

st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');
    /* Global Reset & Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f1116;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    .stApp {
        background: radial-gradient(circle at top, #020617 0%, #020617 35%, #020617 60%, #020617 100%);
        background-color: #020617; /* deep neutral navy */
        font-family: 'Outfit', sans-serif;
        color: #e5e7eb;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        letter-spacing: -0.02em;
    }
    /* Inputs */
    .stTextInput input {
        background-color: #1e293b !important; /* Dark Slate Blue */
        color: #f8fafc !important; /* Bright White */
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        caret-color: #ffffff !important;
        cursor: text !important;
    }
    .stTextInput input:focus {
        background-color: #0f172a !important; /* Darker background on focus */
        border-color: #818cf8 !important; /* Indigo Glow */
        box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.2) !important;
    }
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }
    /* Primary Buttons – deeper indigo to contrast with cyan tabs */
    .stButton button {
        background: linear-gradient(135deg, #4338ca 0%, #4c1d95 100%) !important; /* Indigo → Deep Violet */
        color: #f9fafb !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.6) !important;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.85) !important;
        background: linear-gradient(135deg, #3730a3 0%, #3b0764 100%) !important;
    }
    /* Sidebar "Sign Out" button – subtle filled pill that matches background */
    section[data-testid="stSidebar"] .stButton:nth-of-type(1) > button {
        background: linear-gradient(135deg, #020617 0%, #020617 100%) !important;
        border-radius: 999px !important;
        border: 1px solid #4b5563 !important;
        color: #e5e7eb !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.7) !important;
        padding-inline: 1.8rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stButton:nth-of-type(1) > button:hover {
        background: linear-gradient(135deg, #0b1120 0%, #020617 100%) !important;
        border-color: #9ca3af !important;
        color: #f9fafb !important;
        box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.35) !important;
    }
    /* Cards */
    .book-card {
        background: rgba(30, 41, 59, 0.4); /* More transparent/glassy */
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px; /* Softer corners */
        padding: 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 650px; /* Increased height for better spacing */
        min-height: 650px;
        max-height: 650px;
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Distribute space evenly */
        position: relative;
        overflow: hidden; /* Ensure content stays inside */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem; /* Add spacing between rows if they wrap */
    }
    .book-card:hover {
        transform: translateY(-8px) scale(1.01);
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(139, 92, 246, 0.5); /* Violet glow */
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    .book-card img:hover {
        transform: scale(1.05);
    }
    .book-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #818cf8, transparent);
        opacity: 0;
        transition: opacity 0.4s;
    }
    .book-card:hover::before {
        opacity: 1;
    }
    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    .book-author {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.25rem;
        font-weight: 500;
    }
    /* Hero Section – simple, no boxed container */
    .hero-container {
        position: relative;
        padding: 5rem 2rem 3rem 2rem;
        background: transparent;
        border-radius: 0;
        margin-bottom: 2rem;
        border: none;
        overflow: visible;
        text-align: center;
        box-shadow: none;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e5e7eb 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        animation: fadeIn 1s ease-out;
        text-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero-subtitle {
        font-size: 1.4rem;
        color: #cbd5e1;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
        animation: slideUp 1s ease-out 0.3s backwards;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #a5b4fc; /* soft indigo to match main theme */
        margin-top: 2.5rem;
        margin-bottom: 1.25rem;
        font-weight: 700;
    }
    /* Navigation Tabs (Radio) */
    .stRadio > label {
        display: none !important;
    }
    .stRadio > div[role="radiogroup"] {
        gap: 0.6rem;
        padding: 1rem 0 1.5rem 0;
        display: flex;
        flex-wrap: nowrap;              /* keep tabs on a single line */
        justify-content: center;
        align-items: center;
        overflow-x: auto;               /* allow horizontal scroll on very small screens */
        scrollbar-width: none;          /* Firefox */
    }
    .stRadio > div[role="radiogroup"]::-webkit-scrollbar {
        display: none;                  /* hide scrollbar in WebKit */
    }
    /* Base tab – high-contrast, professional blue pills */
    .stRadio > div[role="radiogroup"] > label {
        background: linear-gradient(135deg, #dbeafe 0%, #60a5fa 45%, #1d4ed8 100%) !important;
        border-radius: 999px !important;
        padding: 0.7rem 1.7rem !important;
        color: #0f172a !important; /* dark text for readability */
        border: 1px solid #bfdbfe !important; /* light blue border */
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em;
        cursor: pointer !important;
        display: flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        min-width: 165px;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease, border-color 0.18s ease, color 0.18s ease !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.7) !important;
        text-shadow: none !important; /* keep text crisp */
    }
    /* Hide default radio circle */
    .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    /* Hover – slightly brighter blue with subtle lift */
    .stRadio > div[role="radiogroup"] > label:hover {
        background: linear-gradient(135deg, #eff6ff 0%, #93c5fd 45%, #2563eb 100%) !important;
        border-color: #e5edff !important;
        color: #0f172a !important;
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 12px 26px rgba(15, 23, 42, 1) !important;
    }
    /* Selected tab – stronger blue/indigo accent, still very readable */
    .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 40%, #4338ca 100%) !important;
        border-color: #bfdbfe !important;
        color: #f9fafb !important; /* white text on active tab */
        box-shadow:
            0 0 0 1px rgba(147, 197, 253, 0.9),
            0 0 18px rgba(37, 99, 235, 0.65),
            0 14px 30px rgba(15, 23, 42, 1) !important;
        transform: translateY(-1px) scale(1.04);
        animation: softPulse 3.8s ease-in-out infinite alternate;
    }
    /* Special styling for first tab ("Trending Now") */
    .stRadio > div[role="radiogroup"] > label:nth-child(1) {
        background: linear-gradient(135deg, #fed7aa 0%, #fb923c 40%, #ea580c 100%) !important; /* warm amber/orange */
        border-color: #fed7aa !important;
        color: #0f172a !important;
    }
    .stRadio > div[role="radiogroup"] > label:nth-child(1):hover {
        background: linear-gradient(135deg, #ffedd5 0%, #fdba74 40%, #f97316 100%) !important;
        border-color: #ffedd5 !important;
        color: #0f172a !important;
    }
    /* Second tab ("Neural Picks") – violet accent */
    .stRadio > div[role="radiogroup"] > label:nth-child(2) {
        background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 40%, #7c3aed 100%) !important;
        border-color: #c7d2fe !important;
        color: #111827 !important;
    }
    .stRadio > div[role="radiogroup"] > label:nth-child(2):hover {
        background: linear-gradient(135deg, #eef2ff 0%, #c4b5fd 40%, #8b5cf6 100%) !important;
        border-color: #e0e7ff !important;
        color: #020617 !important;
    }
    /* Third tab ("Content Matches") – cool cyan */
    .stRadio > div[role="radiogroup"] > label:nth-child(3) {
        background: linear-gradient(135deg, #cffafe 0%, #67e8f9 40%, #06b6d4 100%) !important;
        border-color: #a5f3fc !important;
        color: #022c22 !important;
    }
    .stRadio > div[role="radiogroup"] > label:nth-child(3):hover {
        background: linear-gradient(135deg, #e0f2fe 0%, #7dd3fc 40%, #0ea5e9 100%) !important;
        border-color: #bae6fd !important;
        color: #022c22 !important;
    }
    /* Fourth tab ("Search by Description") – deep blue */
    .stRadio > div[role="radiogroup"] > label:nth-child(4) {
        background: linear-gradient(135deg, #dbeafe 0%, #60a5fa 40%, #1d4ed8 100%) !important;
        border-color: #bfdbfe !important;
        color: #0f172a !important;
    }
    .stRadio > div[role="radiogroup"] > label:nth-child(4):hover {
        background: linear-gradient(135deg, #eff6ff 0%, #93c5fd 40%, #2563eb 100%) !important;
        border-color: #e5edff !important;
        color: #020617 !important;
    }
    /* Fifth tab ("Search by Title") – teal/green accent */
    .stRadio > div[role="radiogroup"] > label:nth-child(5) {
        background: linear-gradient(135deg, #dcfce7 0%, #86efac 40%, #16a34a 100%) !important;
        border-color: #bbf7d0 !important;
        color: #064e3b !important;
    }
    .stRadio > div[role="radiogroup"] > label:nth-child(5):hover {
        background: linear-gradient(135deg, #ecfdf5 0%, #a7f3d0 40%, #22c55e 100%) !important;
        border-color: #d1fae5 !important;
        color: #022c22 !important;
    }
    /* Rating Slider */
    .stSlider > div > div > div > div {
        background-color: #38bdf8 !important; /* Cyan slider */
    }
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2);
    }
    /* Details/Summary */
    /* Details/Summary */
    details {
        background: rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        margin-top: auto;
        transition: all 0.3s ease;
        overflow: hidden;
    }
    details[open] {
        background: rgba(15, 23, 42, 0.6);
        border-color: rgba(255, 255, 255, 0.1);
    }
    summary {
        padding: 0.6rem 1rem;
        cursor: pointer;
        font-weight: 500;
        font-size: 0.85rem;
        color: #94a3b8;
        transition: all 0.2s;
        list-style: none; /* Hide default arrow */
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    summary::-webkit-details-marker {
        display: none; /* Hide default arrow in WebKit */
    }
    summary:hover {
        color: #f8fafc;
        background: rgba(255, 255, 255, 0.03);
    }
    summary::after {
        content: '+';
        font-size: 1.1rem;
        font-weight: 300;
        transition: transform 0.3s;
    }
    details[open] summary::after {
        transform: rotate(45deg);
    }
    .book-desc {
        padding: 0 1rem 1rem 1rem;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #cbd5e1;
        animation: fadeIn 0.3s ease-out;
    }
    /* Section Headers */
    .section-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 3rem 0 1rem 0;
        background: linear-gradient(to right, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .section-desc {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2.5rem;
        font-weight: 300;
        max-width: 700px;
    }
    /* Auth Tabs (Login / Sign Up) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: none;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.9);
        color: #e5e7eb;
        border-radius: 999px;
        padding: 0.4rem 1.4rem;
        font-weight: 500;
        border: 1px solid rgba(51, 65, 85, 0.9);
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(139, 92, 246, 0.2); /* Violet tint */
        border-color: #a78bfa; /* Violet 400 */
        color: #f9fafb;
    }
    .stTabs [aria-selected="true"][data-baseweb="tab"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); /* Violet 500 -> 700 */
        color: #ffffff;
        border-color: #c4b5fd; /* Violet 300 */
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
    }
    /* Auth primary actions – match login/signup tabs */
    .stTabs [data-baseweb="tab-panel"] [data-testid="stFormSubmitButton"] > button,
    .stTabs [data-baseweb="tab-panel"] .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%) !important; /* Violet 500 -> 700 */
        color: #ffffff !important;
        border-radius: 999px !important;
        border: 1px solid #c4b5fd !important; /* Violet 300 */
        box-shadow: 0 10px 22px rgba(139, 92, 246, 0.4) !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-panel"] [data-testid="stFormSubmitButton"] > button:hover,
    .stTabs [data-baseweb="tab-panel"] .stButton > button:hover {
        filter: brightness(1.1);
        transform: translateY(-1px) !important;
        box-shadow: 0 14px 28px rgba(139, 92, 246, 0.6) !important;
    }
    .stTabs [data-baseweb="tab-panel"] [data-testid="stFormSubmitButton"] > button:active,
    .stTabs [data-baseweb="tab-panel"] .stButton > button:active {
        transform: translateY(1px) !important;
    }
    /* Integrated Card Form */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stForm"]:hover {
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.5);
    }
    /* Make the submit button visible and styled like other primary buttons */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4338ca 0%, #4c1d95 100%) !important;
        color: #f9fafb !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        width: 100%;
        margin-top: auto;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #3730a3 0%, #3b0764 100%) !important;
        transform: translateY(-1px);
    }
    </style>
    <script>
    window.addEventListener('load', function() {
        const rootDoc = window.parent && window.parent.document ? window.parent.document : document;
        const inputs = rootDoc.querySelectorAll('input');
        inputs.forEach(function(input) {
            input.setAttribute('autocomplete', 'off');
            input.setAttribute('autocorrect', 'off');
            input.setAttribute('autocapitalize', 'off');
            input.setAttribute('spellcheck', 'false');
        });
    });
    </script>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_and_prep_data():
    books, ratings = load_data()
    valid_ids = get_valid_user_ids()
    ratings, num_users, num_books, user2user_encoded, book2book_encoded = (
        prepare_data_for_nn(ratings)
    )
    return (
        books,
        ratings,
        num_users,
        num_books,
        user2user_encoded,
        book2book_encoded,
        valid_ids,
    )


def render_loading_screen(progress, message):
    st.markdown(
        f"""
        <div style="
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: #0f1116; z-index: 9999;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        ">
            <div style="width: 300px; text-align: center;">
                <div style="font-family: 'Playfair Display', serif; font-size: 2rem; color: #f8fafc; margin-bottom: 1rem;">
                    Preparing Your Library
                </div>
                <div style="font-family: 'Outfit', sans-serif; color: #94a3b8; margin-bottom: 2rem;">
                    {message}
                </div>
                <div style="
                    width: 100%; height: 6px; background: rgba(255,255,255,0.1);
                    border-radius: 3px; overflow: hidden; position: relative;
                ">
                    <div style="
                        width: {int(progress * 100)}%; height: 100%;
                        background: linear-gradient(90deg, #4f46e5, #ec4899);
                        border-radius: 3px; transition: width 0.3s ease;
                        box-shadow: 0 0 10px rgba(79, 70, 229, 0.5);
                    "></div>
                </div>
                <div style="margin-top: 1rem; color: #6366f1; font-weight: 600;">
                    {int(progress * 100)}%
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def train_nn_model(
    ratings, num_users, num_books, _progress_callback=None, force_retrain=False
):
    model_path = os.path.join(os.path.dirname(__file__), "../data/model.pth")
    model = RecommenderNet(num_users, num_books)
    meta_path = os.path.join(os.path.dirname(__file__), "../data/model_metadata.json")
    should_retrain = force_retrain
    if not should_retrain and os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            import json

            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("num_ratings") != len(ratings):
                should_retrain = True
                print(
                    f"Data changed ({meta.get('num_ratings')} -> {len(ratings)} ratings). Retraining..."
                )
        except:
            should_retrain = True
    if not should_retrain and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            if _progress_callback:
                _progress_callback(5, 5)  # Show 100%
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
    dataset = BookDataset(
        ratings["user_encoded"].values,
        ratings["book_encoded"].values,
        ratings["rating"].values,
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_model(model, train_loader, epochs=5, progress_callback=_progress_callback)
    try:
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        print(f"Error saving model: {e}")
    try:
        import json

        with open(meta_path, "w") as f:
            json.dump({"num_ratings": len(ratings)}, f)
    except Exception as e:
        print(f"Error saving metadata: {e}")
    return model


@st.cache_resource
def get_content_model_v4(books):
    return ContentBasedRecommender(books)


def render_book_card(
    title,
    author,
    genres,
    description,
    image_url=None,
    rating=None,
    is_prediction=False,
    explanation=None,
    plain=False,
    avg_rating=None,
    book_id=None,
):
    rating_html = ""
    if rating is not None:
        rating_val = float(rating)
        color = (
            "#10b981"
            if rating_val >= 4.0
            else "#f59e0b" if rating_val >= 3.0 else "#ef4444"
        )
        rating_html = f"""<div class="book-rating"><span style="color: #94a3b8; font-size: 0.75rem;">{'Predicted' if is_prediction else 'Rating'}</span><div class="rating-badge" style="color: {color}"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>{rating_val:.1f}</div></div>"""
    elif explanation:
        rating_html = f"""<div class="book-rating"><span style="color: #94a3b8; font-size: 0.75rem;">Why this book?</span><div class="rating-badge" style="color: #6366f1; font-size: 0.75rem;">{explanation}</div></div>"""
    else:
        rating_html = f"""<div class="book-rating"><span style="color: #94a3b8; font-size: 0.75rem;">Match</span><div class="rating-badge" style="color: #6366f1">Recommended</div></div>"""
    image_html = ""
    if image_url and str(image_url) != "nan":
        image_html = f"""<div style="height: 260px; overflow: hidden; border-radius: 8px; margin-bottom: 1.25rem; display: flex; justify-content: center; align-items: center; background: transparent;"><img src="{image_url}" style="height: 100%; width: auto; object-fit: contain; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3)); transition: transform 0.3s ease;" alt="{title}"></div>"""
    genre_html = ""
    if genres and str(genres) != "nan":
        try:
            clean_genres = (
                str(genres)
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(", ")
            )
            top_genres = clean_genres[:3]
            genre_spans = "".join(
                [
                    f'<span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-right: 6px; display: inline-block; margin-bottom: 4px;">{g}</span>'
                    for g in top_genres
                ]
            )
            genre_html = f'<div style="margin-bottom: 0.75rem;">{genre_spans}</div>'
        except:
            pass
    avg_rating_html = ""
    if avg_rating is not None:
        avg_rating_html = f"""<div style="margin-top: 0.25rem; display: flex; align-items: center; gap: 0.5rem;"><span style="color: #94a3b8; font-size: 0.75rem;">Global Avg:</span><span style="color: #fbbf24; font-size: 0.8rem; font-weight: 600;">⭐ {avg_rating}</span></div>"""
    # Truncate description for the card view
    short_desc = description
    if len(description) > 130:
        short_desc = description[:130].strip() + "..."

    # Updated card content: Rating info is now INSIDE the flex column, at the bottom
    
    # If we have a book_id, we want the IMAGE to be clickable
    if book_id is not None:
        params = f"?book_id={book_id}"
        if "user_id" in st.query_params:
            params += f"&user_id={st.query_params['user_id']}"
        
        # Wrap ONLY the image in the anchor tag
        image_html = f'<a href="{params}" target="_self" style="text-decoration: none; display: block;">{image_html}</a>'

    card_content = f"""{image_html}<div style="flex: 1; display: flex; flex-direction: column;"><div class="book-genre">{author}</div><div class="book-title">{title}</div>{genre_html}{avg_rating_html}<div style="margin-top: 1rem;"><details><summary>View Description</summary><div class="book-desc">{short_desc}</div></details></div><div style="margin-top: auto; padding-top: 1rem;">{rating_html}</div></div>"""
    
    if plain:
        return f'<div style="margin-bottom: 1rem;">{card_content}</div>'
    else:
        return f'<div class="book-card">{card_content}</div>'


def get_user_favorite_genre(user_id, ratings, books):
    user_ratings = ratings[ratings["user_id"] == user_id]
    if user_ratings.empty:
        return None
    user_books = user_ratings.merge(books, on="book_id")
    high_rated = user_books[user_books["rating"] >= 4.0]
    if high_rated.empty:
        high_rated = user_books  # Fallback to all books
    if high_rated.empty:
        return None
    top_genre = high_rated["author"].mode().iloc[0]
    return top_genre


def render_book_details_page(book_id, books, ratings, content_model):
    try:
        book = books[books["book_id"] == int(book_id)].iloc[0]
    except:
        st.error("Book not found.")
        return
    if st.button("← Back to Dashboard"):
        st.query_params.pop("book_id", None)
        st.rerun()
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        if book.get("image_url") and str(book["image_url"]) != "nan":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                    <img src="{book['image_url']}" style="max-height: 450px; width: auto; border-radius: 12px; box-shadow: 0 20px 40px rgba(0,0,0,0.4);">
                </div>
             """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="height: 300px; background: #1e293b; border-radius: 12px; display: flex; align-items: center; justify-content: center; color: #64748b;">No Image Available</div>',
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown(
            f"""
        <div style="margin-bottom: 2rem;">
            <div style="font-size: 1rem; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 0.5rem;">{book['author']}</div>
            <div style="font-size: 3rem; font-weight: 800; color: #f8fafc; line-height: 1.2; margin-bottom: 1rem;">{book['title']}</div>
            <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem;">
                <div style="background: rgba(251, 191, 36, 0.1); color: #fbbf24; padding: 0.4rem 0.8rem; border-radius: 8px; font-weight: 600; border: 1px solid rgba(251, 191, 36, 0.2);">
                    ⭐ {get_book_average_rating(int(book_id))} Average Rating
                </div>
                <div style="color: #94a3b8;">{book.get('year_of_publication', 'N/A')}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("### About this Book")
        st.markdown(
            f'<div style="font-size: 1.1rem; line-height: 1.7; color: #cbd5e1; margin-bottom: 2rem;">{book["description"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Genres")
        if book.get("genres") and str(book["genres"]) != "nan":
            try:
                clean_genres = (
                    str(book["genres"])
                    .replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .split(", ")
                )
                badges_html = "".join(
                    [
                        f"""<span style="
                        display: inline-block;
                        background: rgba(56, 189, 248, 0.1);
                        color: #38bdf8;
                        padding: 0.5rem 1rem;
                        border-radius: 999px;
                        font-size: 0.85rem;
                        font-weight: 600;
                        margin-right: 0.5rem;
                        margin-bottom: 0.5rem;
                        border: 1px solid rgba(56, 189, 248, 0.2);
                        transition: all 0.2s ease;
                    ">{g}</span>"""
                        for g in clean_genres
                    ]
                )
                st.markdown(
                    f'<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">{badges_html}</div>',
                    unsafe_allow_html=True,
                )
            except:
                pass
    st.markdown("---")
    st.markdown("### You might also like")
    recs = content_model.get_recommendations(book["title"])
    cols = st.columns(4)
    for i, (rec_title, score) in enumerate(recs[:4]):
        rec_book = books[books["title"] == rec_title].iloc[0]
        with cols[i]:
            st.markdown(
                render_book_card(
                    rec_book["title"],
                    rec_book["author"],
                    rec_book.get("genres"),
                    rec_book["description"],
                    rec_book.get("image_url"),
                    explanation=f"{int(score*100)}% Match",
                    book_id=rec_book["book_id"],
                ),
                unsafe_allow_html=True,
            )


def login_page(valid_user_ids):
    st.markdown(
        """
        <div class="hero-container" style="padding: 3rem 1rem; margin-bottom: 2rem; background: transparent; box-shadow: none; border: none;">
            <div class="hero-title" style="font-size: 3.5rem; margin-bottom: 0.5rem;">Welcome Back</div>
            <div class="hero-subtitle">Sign in to access your personalized library</div>
        </div>
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        with tab_login:
            with st.form("login_form"):
                st.markdown(
                    '<label style="color: #e2e8f0; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; display: block;">User ID</label>',
                    unsafe_allow_html=True,
                )
                user_id_input = st.text_input(
                    "User ID",
                    label_visibility="collapsed",
                    placeholder="Enter your User ID",
                )
                st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In")
                if submit:
                    try:
                        user_id = int(user_id_input)
                        if user_id in valid_user_ids:
                            st.session_state.loading_in_progress = True
                            st.session_state.loading_user_id = user_id
                            st.rerun()
                        else:
                            st.error("Invalid User ID. Please try again.")
                    except ValueError:
                        st.error("Please enter a valid numeric User ID.")
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            new_username = st.text_input(
                "Choose a Username (Optional)", placeholder="Guest"
            )
            if st.button("Create Account", type="primary"):
                new_id = create_user(
                    new_username
                    if new_username
                    else f"User_{np.random.randint(1000,9999)}"
                )
                if new_id:
                    st.success(f"Account Created! Your User ID is: **{new_id}**")
                    st.info("Please memorize this ID to log in.")
                    st.cache_resource.clear()
                else:
                    st.error("Error creating account.")


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    with st.spinner("Loading System..."):
        (
            books,
            ratings,
            num_users,
            num_books,
            user2user_encoded,
            book2book_encoded,
            valid_user_ids,
        ) = load_and_prep_data()
    if "book_id" in st.query_params:
        book_id = st.query_params["book_id"]
        content_model = get_content_model_v4(books)
        render_book_details_page(book_id, books, ratings, content_model)
        return
    if st.session_state.get("loading_in_progress"):


        def update_progress(current, total):
            p = 0.2 + (current / total) * 0.7
            render_loading_screen(
                p, f"Fine-tuning Neural Network (Epoch {current}/{total})..."
            )

        render_loading_screen(0.1, "Initializing User Profile...")
        time.sleep(0.3)  # Small delay for visual smoothness
        render_loading_screen(0.2, "Checking for Saved Brain...")
        time.sleep(0.3)
        force_retrain = st.session_state.get("force_retrain", False)
        if force_retrain:
            st.session_state.force_retrain = False  # Reset flag
            train_nn_model(
                ratings,
                num_users,
                num_books,
                _progress_callback=update_progress,
                force_retrain=True,
            )
        else:
            model = train_nn_model(ratings, num_users, num_books, force_retrain=False)
            for i in range(20, 91, 5):
                p = i / 100.0
                render_loading_screen(p, "Loading Neural Network...")
                time.sleep(0.05)  # Fast but visible smooth fill
        render_loading_screen(0.9, "Finalizing Content Models...")
        get_content_model_v4(books)
        time.sleep(0.3)
        render_loading_screen(1.0, "Welcome to Your Library!")
        time.sleep(0.5)
        st.session_state.logged_in = True
        st.session_state.user_id = st.session_state.loading_user_id
        del st.session_state.loading_in_progress
        del st.session_state.loading_user_id
        st.rerun()
        return
    if not st.session_state.logged_in:
        params = st.query_params
        if "user_id" in params:
            try:
                uid = int(params["user_id"])
                if uid in valid_user_ids:
                    st.session_state.logged_in = True
                    st.session_state.user_id = uid
            except:
                pass
    if not st.session_state.logged_in:
        login_page(valid_user_ids)
    else:
        st.query_params["user_id"] = str(st.session_state.user_id)
        with st.spinner("Fine-tuning Neural Network..."):
            nn_model = train_nn_model(ratings, num_users, num_books)
        with st.spinner("Analyzing Content Features..."):
            content_model = get_content_model_v4(books)
        st.markdown(
            """
        <div class="hero-container">
            <div class="hero-title">Discover Your Next Great Read</div>
            <div class="hero-subtitle">Our AI-powered engine analyzes your reading patterns to curate a personalized library just for you.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        with st.sidebar:
            st.markdown(
                f'<div class="sidebar-header">User Profile</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 2rem;">
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">Logged in as</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">User #{st.session_state.user_id}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.button("Sign Out"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.query_params.clear()
                st.rerun()
            st.markdown("---")
            stats = get_system_stats()
            st.markdown(
                '<div class="sidebar-header">Statistics</div>', unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Books", f"{stats['num_books']:,}")
            with c2:
                st.metric("Users", f"{stats['num_users']:,}")
            st.metric("Total Ratings", f"{stats['num_ratings']:,}")
            st.markdown("---")
            st.markdown(
                '<div style="font-size: 0.75rem; color: #64748b; text-align: center;">Powered by PyTorch & Streamlit</div>',
                unsafe_allow_html=True,
            )
        options = [
            "🔥 Trending Now",
            "🧠 Neural Picks",
            "🔍 Content Matches",
            "📝 Search by Description",
            "🔎 Search by Title",
        ]
        if "nav_tab" not in st.session_state:
            default_tab = options[0]
            if "tab" in st.query_params:
                if st.query_params["tab"] in options:
                    default_tab = st.query_params["tab"]
            st.session_state.nav_tab = default_tab

        def update_tab_params():
            st.query_params["tab"] = st.session_state.nav_tab

        selected_tab = st.radio(
            "Navigation",
            options,
            key="nav_tab",
            horizontal=True,
            label_visibility="collapsed",
            on_change=update_tab_params,
        )
        st.markdown("---")
        user_id = st.session_state.user_id
        if selected_tab == "🔥 Trending Now":
            st.markdown(
                '<div class="section-header"><span>🔥</span> Trending Now</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-desc">The top 10 most popular books in our community right now.</div>',
                unsafe_allow_html=True,
            )
            top_book_ids = ratings["book_id"].value_counts().head(10).index
            cols = st.columns(3, gap="large")
            for i, book_id in enumerate(top_book_ids):
                book_info = books[books["book_id"] == book_id].iloc[0]
                col_idx = i % 3
                live_avg = get_book_average_rating(book_id)
                with cols[col_idx]:
                    st.markdown(
                        render_book_card(
                            book_info["title"],
                            book_info["author"],
                            book_info.get("genres"),
                            book_info["description"],
                            book_info.get("image_url"),
                            rating=None,
                            avg_rating=live_avg,
                            explanation=f"#{i+1} Popular",
                            book_id=book_id,
                        ),
                        unsafe_allow_html=True,
                    )
        elif selected_tab == "🧠 Neural Picks":
            st.markdown(
                '<div class="section-header"><span>🧠</span> Neural Engine Picks</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-desc">Deep learning predictions based on your unique reading history and similar users.</div>',
                unsafe_allow_html=True,
            )
            user_encoded = user2user_encoded.get(user_id)
            if user_encoded is not None:
                all_book_ids = torch.tensor(
                    list(book2book_encoded.values()), dtype=torch.long
                )
                user_tensor = torch.tensor(
                    [user_encoded] * len(all_book_ids), dtype=torch.long
                )
                nn_model.eval()
                with torch.no_grad():
                    predictions = nn_model(user_tensor, all_book_ids).squeeze()
                top_indices = predictions.argsort(descending=True)[:3].numpy()
                encoded2book = {v: k for k, v in book2book_encoded.items()}
                fav_genre = get_user_favorite_genre(user_id, ratings, books)
                explanation_text = (
                    f"Matches your interest in {fav_genre}"
                    if fav_genre
                    else "Trending among similar users"
                )
                cols = st.columns(3)
                for i, idx in enumerate(top_indices):
                    book_id = encoded2book[idx]
                    book_info = books[books["book_id"] == book_id].iloc[0]
                    live_avg = get_book_average_rating(int(book_id))
                    with cols[i]:
                        st.markdown(
                            render_book_card(
                                book_info["title"],
                                book_info["author"],
                                book_info.get("genres"),
                                book_info["description"],
                                book_info.get("image_url"),
                                predictions[idx],
                                is_prediction=True,
                                explanation=explanation_text,
                                avg_rating=live_avg,
                                book_id=book_id,
                            ),
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("New user? No history found.")
        elif selected_tab == "🔍 Content Matches":
            st.markdown(
                '<div class="section-header"><span>🔍</span> Content Matches</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-desc">Books with similar themes and narratives to your favorites.</div>',
                unsafe_allow_html=True,
            )
            user_ratings = ratings[ratings["user_id"] == user_id]
            if not user_ratings.empty:
                top_book_id = user_ratings.sort_values("rating", ascending=False).iloc[
                    0
                ]["book_id"]
                top_book_title = books[books["book_id"] == top_book_id]["title"].values[
                    0
                ]
                st.info(f"Because you enjoyed **{top_book_title}**")
                recs_with_scores = content_model.get_recommendations(top_book_title)
                recs_with_scores = recs_with_scores[:3]
                cols = st.columns(3)
                for i, (rec_title, score) in enumerate(recs_with_scores):
                    book_info = books[books["title"] == rec_title].iloc[0]
                    match_percentage = f"{int(score * 100)}% Match"
                    live_avg = get_book_average_rating(int(book_info["book_id"]))
                    with cols[i]:
                        st.markdown(
                            render_book_card(
                                book_info["title"],
                                book_info["author"],
                                book_info.get("genres"),
                                book_info["description"],
                                book_info.get("image_url"),
                                explanation=match_percentage,
                                avg_rating=live_avg,
                                book_id=book_info["book_id"],
                            ),
                            unsafe_allow_html=True,
                        )
            else:
                st.write("Rate some books to get content-based recommendations!")
        elif selected_tab == "📝 Search by Description":
            st.markdown(
                '<div class="section-header"><span>📝</span> Find by Description</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-desc">Describe what you want to read and press Enter to search.</div>',
                unsafe_allow_html=True,
            )

            def _trigger_desc_search():
                st.session_state["run_desc_search"] = True

            search_query = st.text_input(
                "Enter a description (e.g., 'A mystery about a detective in London')",
                value=st.session_state.get("desc_query", ""),
                key="desc_query",
                on_change=_trigger_desc_search,
                placeholder="Type what you feel like reading and hit Enter...",
            )
            if st.session_state.get("run_desc_search") and search_query:
                with st.spinner("Searching books..."):
                    if not hasattr(content_model, "recommend_by_description"):
                        st.cache_resource.clear()
                        content_model = get_content_model_v4(books)
                    desc_recs = content_model.recommend_by_description(
                        search_query, top_n=3
                    )
                st.session_state["run_desc_search"] = False
                if desc_recs:
                    cols = st.columns(3)
                    for i, (rec_title, score) in enumerate(desc_recs):
                        book_info = books[books["title"] == rec_title].iloc[0]
                        match_percentage = f"{int(score * 100)}% Match"
                        live_avg = get_book_average_rating(int(book_info["book_id"]))
                        with cols[i]:
                            st.markdown(
                                render_book_card(
                                    book_info["title"],
                                    book_info["author"],
                                    book_info.get("genres"),
                                    book_info["description"],
                                    book_info.get("image_url"),
                                    explanation=match_percentage,
                                    avg_rating=live_avg,
                                    book_id=book_info["book_id"],
                                ),
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No matches found. Try a different description.")
        elif selected_tab == "🔎 Search by Title":
            st.markdown(
                '<div class="section-header"><span>🔎</span> Search by Title</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-desc">Search for books and rate them to improve your recommendations.</div>',
                unsafe_allow_html=True,
            )
            rate_search = st.text_input(
                "Search for a book by title",
                placeholder="Enter book title...",
                key="title_search",
            )
            if rate_search:
                results = books[
                    books["title"].str.contains(rate_search, case=False, na=False)
                ].head(50)
                if not results.empty:
                    st.write(f"Found {len(results)} books matching '{rate_search}':")
                    user_ratings_map = get_user_ratings(user_id)
                    cols = st.columns(3, gap="large")
                    for i, (_, book) in enumerate(results.iterrows()):
                        col_idx = i % 3
                        book_id = int(book["book_id"])
                        existing_rating = user_ratings_map.get(book_id)
                        avg_rating = get_book_average_rating(book_id)
                        with cols[col_idx]:
                            with st.form(f"rating_form_{book_id}"):
                                st.markdown(
                                    render_book_card(
                                        book["title"],
                                        book["author"],
                                        book.get("genres"),
                                        book["description"],
                                        book.get("image_url"),
                                        plain=True,
                                        avg_rating=avg_rating,
                                        book_id=book_id,
                                    ),
                                    unsafe_allow_html=True,
                                )
                                
                                st.markdown("---")
                                if existing_rating:
                                    st.markdown(f"<div style='text-align: center; color: #fbbf24; margin-bottom: 0.5rem;'>Current Rating: <b>{existing_rating}</b> ⭐</div>", unsafe_allow_html=True)
                                    default_val = int(existing_rating)
                                    btn_label = "Update Rating"
                                else:
                                    st.markdown("<div style='text-align: center; color: #94a3b8; margin-bottom: 0.5rem;'>Rate this book</div>", unsafe_allow_html=True)
                                    default_val = 5
                                    btn_label = "Submit Rating"
                                
                                rating_val = st.slider(
                                    "Stars", 1, 5, default_val, key=f"slider_{book_id}", label_visibility="collapsed"
                                )
                                submit_rating = st.form_submit_button(btn_label)
                                
                                if submit_rating:
                                    add_rating(user_id, book_id, rating_val)
                                    st.cache_resource.clear()
                                    st.success(f"Saved!")
                                    time.sleep(0.5)
                                    st.rerun()
                else:
                    st.info("No books found matching your search.")


if __name__ == "__main__":
    main()
