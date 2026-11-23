# 🎬 Movie Recommendation System | NLP & Streamlit

A content-based Movie Recommendation System developed using Python, Natural Language Processing (NLP), TF-IDF Vectorization, and Cosine Similarity. The project provides an interactive web interface using Streamlit to recommend movies based on user preferences.

## 🚀 Project Overview

This application recommends similar movies based on the movie selected by the user. It analyzes movie features using NLP techniques and calculates similarity scores to find relevant recommendations.

## ✨ Key Features

- Interactive Streamlit web application
- Content-based movie recommendations
- NLP-based text processing
- TF-IDF Vectorization
- Cosine Similarity
- Movie similarity analysis
- Preprocessed data and saved model files

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Streamlit | Web application |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| NLP | Text processing |
| TF-IDF | Text vectorization |
| Cosine Similarity | Similarity calculation |

## 📂 Project Structure

```text
Movie-Recommendation-System/
│
├── main.py
├── df.pkl
├── indices.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/nancypatelfintech/Movie-Recommendation-System.git
```

### 2. Open the project folder

```bash
cd Movie-Recommendation-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run main.py
```

### 5. Open in browser

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

## 🧠 How the Recommendation System Works

1. Load and preprocess movie datasets.
2. Extract relevant textual features.
3. Convert text into numerical vectors using TF-IDF.
4. Calculate similarity using Cosine Similarity.
5. Find movies with the highest similarity scores.
6. Display recommended movies through the Streamlit interface.

## 📚 Skills Demonstrated

- Python Programming
- Natural Language Processing
- Machine Learning Fundamentals
- Content-Based Recommendation Systems
- Data Preprocessing
- Streamlit Application Development
- Model Serialization using Pickle

## 👨‍💻 Author

**Nancy Patel**

GitHub: https://github.com/nancypatelfintech

## ⭐ Support

If you find this project interesting, consider giving this repository a star!