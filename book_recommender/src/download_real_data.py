import os
import requests
import pandas as pd
import numpy as np

def download_real_data():
    """
    Downloads the 'Best Books Ever' dataset from GitHub.
    Generates synthetic user ratings for these real books to support the Neural Network.
    """
    print("Downloading real book data...")
    url = "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/master/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../data')
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Download Books
        df = pd.read_csv(url, on_bad_lines='skip')
        
        # Select relevant columns and rename to match our schema
        # Schema: bookId, title, series, author, rating, description, language, isbn, genres, characters, bookFormat, edition, pages, publisher, publishDate, firstPublishDate, awards, numRatings, ratingsByStars, likedPercent, setting, coverImg, bbeScore, bbeVotes, price
        
        books_df = df[['bookId', 'title', 'author', 'description', 'genres', 'coverImg', 'rating']].copy()
        books_df = books_df.rename(columns={
            'bookId': 'book_id',
            'coverImg': 'image_url',
            'rating': 'average_rating'
        })
        
        # Clean data
        books_df = books_df.dropna(subset=['title', 'description'])
        # Limit to top 1000 books to keep app fast for demo
        books_df = books_df.head(1000)
        
        # Save books
        books_path = os.path.join(data_dir, 'books.csv')
        books_df.to_csv(books_path, index=False)
        print(f"Saved {len(books_df)} real books to {books_path}")
        
        # Generate Synthetic Ratings for these Real Books
        # (We need this because the dataset doesn't have user-item interaction history)
        print("Generating synthetic user ratings for these books...")
        num_users = 500
        num_ratings = 10000
        
        ratings = []
        book_ids = books_df['book_id'].tolist()
        
        for _ in range(num_ratings):
            user_id = np.random.randint(0, num_users)
            book_id = np.random.choice(book_ids)
            
            # Bias rating towards the book's actual average rating
            # Get book's avg rating (e.g. 4.5)
            avg_rating = books_df.loc[books_df['book_id'] == book_id, 'average_rating'].iloc[0]
            try:
                avg_rating = float(avg_rating)
            except:
                avg_rating = 4.0
                
            # Generate rating with some noise
            rating = int(np.clip(np.random.normal(avg_rating, 1.0), 1, 5))
            
            ratings.append({
                'user_id': user_id,
                'book_id': book_id,
                'rating': rating
            })
            
        ratings_df = pd.DataFrame(ratings)
        ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'book_id'])
        
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        ratings_df.to_csv(ratings_path, index=False)
        print(f"Saved {len(ratings_df)} synthetic ratings to {ratings_path}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_real_data()
