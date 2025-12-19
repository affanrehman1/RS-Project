# 📚 Book Recommendation System

A modern, intelligent **Book Recommendation System** that helps users discover books tailored to their interests using **neural networks** and **content-based filtering**. The system supports searching by **book title** and **description**, and delivers personalized recommendations through a clean, interactive **Streamlit** interface.

---

## 🚀 Features

* 🤖 **Neural Network–based Recommendations**
  Learns meaningful representations of books to improve recommendation quality.

* 🧠 **Content-Based Filtering**
  Recommends books based on similarity in descriptions, genres, and metadata.

* 🔍 **Search Functionality**

  * Search books by **title**
  * Search books by **description / keywords**

* 💻 **Interactive Frontend**
  Built with **Streamlit** for a modern, responsive, and user-friendly experience.

* 🗄️ **Lightweight Database**
  Uses **SQLite3** for simplicity, portability, and easy setup.

---

## 🛠️ Tech Stack

| Layer       | Technology              |
| ----------- | ----------------------- |
| Frontend    | Streamlit               |
| Backend     | Python                  |
| ML Model    | Neural Network          |
| Recommender | Content-Based Filtering |
| Database    | SQLite3                 |

---

## 📂 Project Overview

The system analyzes book descriptions and related metadata to generate meaningful vector representations using a neural network. When a user searches for a book or enters a description, the system computes similarity scores and recommends the most relevant books.

The focus of the project is **accuracy**, **simplicity**, and **real-world usability**.

---

## ⚙️ How It Works

1. User searches for a book by **title** or **description**
2. Text data is processed and converted into feature vectors
3. A neural network learns latent representations of books
4. Similarity is computed between books
5. Top relevant books are recommended to the user

---

## ▶️ Running the Project

Follow these steps to run the project on **any machine (Windows, Linux, macOS)**.

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd book_recommender
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

All required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run src/app.py
```

The application will automatically open in your default web browser.

---

## 📁 Project Structure

The project follows a **single, clean, and consistent directory structure** as shown below:

```
book_recommender/
│
├── data/
│   ├── books.csv         # Book metadata
│   ├── ratings.csv       # User ratings data
│   ├── library.db        # SQLite3 database
│   └── model.pth         # Trained neural network model
│
├── src/
│   ├── app.py            # Streamlit application entry point
│   ├── content_based.py  # Content-based recommendation logic
│   ├── database.py       # Database connection and queries
│   ├── neural_network.py # Neural network model
│   └── preprocessing.py # Data preprocessing
│
├── requirements.txt      # Python dependencies
├── README.md
├── .gitignore
└── Run Code.txt          # Quick run instructions
```

---

## 📄 License

This project is intended for **educational and academic purposes only**.

You are free to use, modify, and extend this project for learning, coursework, and research.

---

## 🧩 Portability & Compatibility

* Uses **relative paths** to ensure the project runs on any system
* Database powered by **SQLite3** (no external DB setup required)
* Tested with **Streamlit** on Windows, Linux, and macOS

---

## 📌 Use Cases

* Personalized book discovery
* Academic and research projects
* Learning recommender systems and neural networks
* Lightweight recommendation engines

---

## 👨‍💻 Team Members

* **Ahsan Faizan**
* **Affan Rehman**
* **Mujtaba Khan**

---

## 📄 License

This project is intended for **educational and academic purposes only**.

You are free to use, modify, and extend this project for learning, coursework, and research.
