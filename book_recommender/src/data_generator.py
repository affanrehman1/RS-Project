import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()

def generate_data(num_books=100, num_users=50, num_ratings=1000):
    print("Generating synthetic data...")
    
    # 1. Generate Books
    genres = ['Fiction', 'Sci-Fi', 'Mystery', 'Romance', 'Thriller', 'Biography', 'History', 'Fantasy']
    books = []
    for i in range(num_books):
        books.append({
            'book_id': i,
            'title': fake.sentence(nb_words=4).replace('.', ''),
            'genre': random.choice(genres),
            'description': fake.paragraph(nb_sentences=3)
        })
    books_df = pd.DataFrame(books)
    
    # 2. Generate Users
    users = []
    for i in range(num_users):
        users.append({
            'user_id': i,
            'name': fake.name()
        })
    users_df = pd.DataFrame(users)
    
    # 3. Generate Ratings
    ratings = []
    for _ in range(num_ratings):
        user_id = random.randint(0, num_users - 1)
        book_id = random.randint(0, num_books - 1)
        rating = random.randint(1, 5)
        
        # Simple logic to make data slightly non-random (e.g., users prefer certain genres)
        # This helps the model learn something meaningful
        user_pref_genre = books_df.iloc[book_id]['genre']
        if random.random() > 0.7: # 30% chance to bias rating based on genre
             # Assign a "favorite" genre to a user implicitly by boosting rating if it matches a random choice
             # For simplicity, let's just say higher ratings for 'Fiction' and 'Sci-Fi' generally
             if user_pref_genre in ['Fiction', 'Sci-Fi']:
                 rating = min(5, rating + 1)
        
        ratings.append({
            'user_id': user_id,
            'book_id': book_id,
            'rating': rating
        })
    ratings_df = pd.DataFrame(ratings)
    
    # Ensure no duplicates
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'book_id'])

    # Save to CSV
    # Use absolute path relative to this script to ensure it goes to book_recommender/data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../data')
    os.makedirs(data_dir, exist_ok=True)
    
    books_df.to_csv(os.path.join(data_dir, 'books.csv'), index=False)
    ratings_df.to_csv(os.path.join(data_dir, 'ratings.csv'), index=False)
    
    print(f"Generated {len(books_df)} books, {len(users_df)} users, and {len(ratings_df)} ratings.")
    print(f"Data saved to {data_dir}")

if __name__ == "__main__":
    generate_data()
