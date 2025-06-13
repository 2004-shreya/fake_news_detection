# 📰 Fake News Detection using Python

This project uses machine learning techniques to classify news articles as *Real* or *Fake*. It applies Natural Language Processing (NLP) to analyze news text and make predictions based on learned patterns.

---

## 📌 Project Objective

The goal of this project is to build a model that can detect fake news from real news articles. The model is trained on a labeled dataset of news content using Python and NLP libraries.

---

## 📁 Dataset

The dataset used is the [Fake and Real News Dataset](https://www.kaggle.com/datasets/antonioskokiantonis/newscsv) from Kaggle, containing two CSV files: one for fake news and one for real news.

---

🧠 Model Overview

The model uses:

Text Preprocessing with NLTK (stopwords removal, stemming)

TF-IDF Vectorization to convert text into numerical form

Machine Learning Models like:

Logistic Regression

Naive Bayes

Passive Aggressive Classifier

---

🚀 How to Run

To train and test the model:

python fake_news_detector.py

To run the Streamlit web app:

streamlit run app.py

---

✅ Features

Binary classification (Real or Fake)

Cleaned and preprocessed data

TF-IDF based text vectorization

Interactive web interface using Streamlit

---

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

NLTK

Streamlit

Jupyter Notebook (for exploration)
