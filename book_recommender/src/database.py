import sqlite3
import pandas as pd
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/library.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database and populates it if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_id TEXT, -- Stores ISBN or original ID
        title TEXT,
        author TEXT,
        year INTEGER,
        publisher TEXT,
        image_url TEXT,
        description TEXT,
        genres TEXT,
        average_rating REAL DEFAULT 0.0
    )
    """
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS ratings (
        rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        book_id INTEGER,
        rating INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (book_id) REFERENCES books (book_id)
    )
    """
    )
    conn.commit()
    cursor.execute("SELECT count(*) FROM books")
    if cursor.fetchone()[0] == 0:
        print("Database empty. Importing data from CSVs...")
        import_data(conn)
    conn.close()


def import_data(conn):
    """Imports data from CSVs into SQLite, handling different schemas."""
    books_path = os.path.join(DATA_DIR, "books.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    users_path = os.path.join(DATA_DIR, "users.csv")  # Optional
    if not os.path.exists(books_path):
        print("No books.csv found!")
        return
    df_books = pd.read_csv(books_path, nrows=5, on_bad_lines="skip")
    columns = df_books.columns.tolist()
    print("Importing Books...")
    if "ISBN" in columns and "Book-Title" in columns:
        real_books = pd.read_csv(
            books_path, on_bad_lines="skip", dtype={"Year-Of-Publication": str}
        )
        real_books = real_books.rename(
            columns={
                "ISBN": "original_id",
                "Book-Title": "title",
                "Book-Author": "author",
                "Year-Of-Publication": "year",
                "Publisher": "publisher",
                "Image-URL-L": "image_url",
            }
        )
        real_books["description"] = "No description available."
        real_books["genres"] = "['General']"
        real_books["average_rating"] = 0.0  # Will be calculated from ratings
        real_books["year"] = (
            pd.to_numeric(real_books["year"], errors="coerce").fillna(0).astype(int)
        )
        db_books = real_books[
            [
                "original_id",
                "title",
                "author",
                "year",
                "publisher",
                "image_url",
                "description",
                "genres",
                "average_rating",
            ]
        ]
        db_books.to_sql("books", conn, if_exists="append", index=False)
        print(f"Imported {len(db_books)} books (Kaggle Schema).")
    elif "book_id" in columns and "title" in columns:
        real_books = pd.read_csv(books_path)
        real_books = real_books.rename(columns={"book_id": "original_id"})
        real_books["year"] = 0
        real_books["publisher"] = "Unknown"
        db_books = real_books[
            [
                "original_id",
                "title",
                "author",
                "year",
                "publisher",
                "image_url",
                "description",
                "genres",
                "average_rating",
            ]
        ]
        db_books.to_sql("books", conn, if_exists="append", index=False)
        print(f"Imported {len(db_books)} books (Goodreads Schema).")
    print("Importing Ratings...")
    if os.path.exists(ratings_path):
        df_ratings = pd.read_csv(ratings_path, nrows=5)
        r_columns = df_ratings.columns.tolist()
        book_map = pd.read_sql("SELECT original_id, book_id FROM books", conn)
        id_map = dict(zip(book_map["original_id"].astype(str), book_map["book_id"]))
        if "User-ID" in r_columns and "ISBN" in r_columns:
            chunk_size = 100000
            for chunk in pd.read_csv(
                ratings_path, chunksize=chunk_size, on_bad_lines="skip"
            ):
                chunk = chunk.rename(
                    columns={
                        "User-ID": "user_id",
                        "ISBN": "original_id",
                        "Book-Rating": "rating",
                    }
                )
                chunk["original_id"] = chunk["original_id"].astype(str)
                chunk["book_id"] = chunk["original_id"].map(id_map)
                chunk = chunk.dropna(
                    subset=["book_id"]
                )  # Drop ratings for missing books
                chunk[["user_id", "book_id", "rating"]].to_sql(
                    "ratings", conn, if_exists="append", index=False
                )
                print(f"Imported chunk of ratings...")
        elif "user_id" in r_columns and "book_id" in r_columns:
            real_ratings = pd.read_csv(ratings_path)
            real_ratings["book_id"] = real_ratings["book_id"].astype(str).map(id_map)
            real_ratings = real_ratings.dropna(subset=["book_id"])
            real_ratings[["user_id", "book_id", "rating"]].to_sql(
                "ratings", conn, if_exists="append", index=False
            )
            print(f"Imported {len(real_ratings)} ratings.")
    print("Syncing Users...")
    conn.execute(
        """
        INSERT OR IGNORE INTO users (user_id, username)
        SELECT DISTINCT user_id, 'User ' || user_id FROM ratings
    """
    )
    conn.commit()
    print("Data Import Complete.")


def get_all_books():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM books", conn)
    conn.close()
    return df


def get_all_ratings():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM ratings", conn)
    conn.close()
    return df


def create_user(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_username(username):
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return user


def add_rating(user_id, book_id, rating):
    conn = get_db_connection()
    cursor = conn.cursor()
    existing = cursor.execute(
        "SELECT rating_id FROM ratings WHERE user_id = ? AND book_id = ?",
        (user_id, book_id),
    ).fetchone()
    if existing:
        cursor.execute(
            "UPDATE ratings SET rating = ?, timestamp = CURRENT_TIMESTAMP WHERE rating_id = ?",
            (rating, existing["rating_id"]),
        )
    else:
        cursor.execute(
            "INSERT INTO ratings (user_id, book_id, rating) VALUES (?, ?, ?)",
            (user_id, book_id, rating),
        )
    conn.commit()
    conn.close()


def get_valid_user_ids():
    conn = get_db_connection()
    df = pd.read_sql("SELECT user_id FROM users", conn)
    conn.close()
    return df["user_id"].tolist()


def get_user_ratings(user_id):
    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT book_id, rating FROM ratings WHERE user_id = ?", conn, params=(user_id,)
    )
    conn.close()
    if df.empty:
        return {}
    return dict(zip(df["book_id"], df["rating"]))


def get_system_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    num_books = cursor.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    num_users = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    num_ratings = cursor.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    conn.close()
    return {"num_books": num_books, "num_users": num_users, "num_ratings": num_ratings}


def get_book_average_rating(book_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    avg_rating = cursor.execute(
        "SELECT AVG(rating) FROM ratings WHERE book_id = ?", (book_id,)
    ).fetchone()[0]
    conn.close()
    if avg_rating is None:
        return 0.0
    return round(avg_rating, 1)
