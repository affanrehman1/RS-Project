import pandas as pd
import os

def load_data():
    """Loads books and ratings data."""
    data_path = os.path.join(os.path.dirname(__file__), '../data')
    books = pd.read_csv(os.path.join(data_path, 'books.csv'))
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    return books, ratings

def prepare_data_for_nn(ratings_df):
    """Encodes user and book IDs for the neural network."""
    # Filter ratings to only include books that exist in our books dataframe
    # (This is handled in the main app logic usually, but good to be safe)
    
    user_ids = ratings_df['user_id'].unique().tolist()
    book_ids = ratings_df['book_id'].unique().tolist()
    
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    
    ratings_df['user_encoded'] = ratings_df['user_id'].map(user2user_encoded)
    ratings_df['book_encoded'] = ratings_df['book_id'].map(book2book_encoded)
    
    num_users = len(user2user_encoded)
    num_books = len(book2book_encoded)
    
    return ratings_df, num_users, num_books, user2user_encoded, book2book_encoded
