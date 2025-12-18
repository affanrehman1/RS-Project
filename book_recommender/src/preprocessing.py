from database import get_all_books, get_all_ratings, init_db


def load_data():
    """Loads books and ratings data from the database."""
    init_db()
    books = get_all_books()
    ratings = get_all_ratings()
    return books, ratings


def prepare_data_for_nn(ratings_df):
    """Encodes user and book IDs for the neural network."""
    user_ids = ratings_df["user_id"].unique().tolist()
    book_ids = ratings_df["book_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    ratings_df["user_encoded"] = ratings_df["user_id"].map(user2user_encoded)
    ratings_df["book_encoded"] = ratings_df["book_id"].map(book2book_encoded)
    num_users = len(user2user_encoded)
    num_books = len(book2book_encoded)
    return ratings_df, num_users, num_books, user2user_encoded, book2book_encoded
