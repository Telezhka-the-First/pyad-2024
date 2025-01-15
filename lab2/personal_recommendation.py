"""## 4. В файле personal_recommendation.py создайте рекомендацию для пользователя, у которого в исходном датасете было больше всего 0 среди рейтингов книг. Алгоритм такой:

### 0. Загружаем библиотеки и модели, обрабатываем книги.
"""

import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack
import numpy as np

ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv")

with open("svd.pkl", "rb") as file:
    svd = pickle.load(file)

with open("linreg.pkl", "rb") as file:
    linreg = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("author_encoder.pkl", "rb") as file:
    author_encoder = pickle.load(file)

with open("publisher_encoder.pkl", "rb") as file:
    publisher_encoder = pickle.load(file)

with open("year_scaler.pkl", "rb") as file:
    year_scaler = pickle.load(file)

books["Year-Of-Publication"] = books["Year-Of-Publication"].astype(str)

shift_start_index = books.columns.get_loc("Book-Author")

def shift_row(row):
    if not row["Year-Of-Publication"].isdigit():

        row.iloc[shift_start_index + 1:] = row.iloc[shift_start_index:-1].values
        row.iloc[shift_start_index] = np.nan

        if "\\\";" in row["Book-Title"]:
            title_parts = row["Book-Title"].split('\\\";', 1)
            row["Book-Title"] = title_parts[0].strip()
            row["Book-Author"] = title_parts[1].strip(' "')
    return row

books = books.apply(shift_row, axis=1)
books = books[books["Year-Of-Publication"].astype(int) <= 2025]
books.dropna(inplace=True)
books.drop(columns=[col for col in books.columns if col.startswith("Image-URL-")], inplace=True)

"""### 1. Находим нужного пользователя."""

user_with_most_zeros = ratings[ratings["Book-Rating"] == 0]["User-ID"].value_counts().idxmax()

"""### 2. Делаем предсказание SVD для книг, которым он "поставил" 0."""

zero_rated_books = ratings[(ratings["User-ID"] == user_with_most_zeros) & (ratings["Book-Rating"] == 0)]["ISBN"].unique()

"""### 3. Берем те книги, для которых предсказали рейтинг не ниже 8. Считаем, что 8 означет, что книга ему точно понравится."""

def predict_rating(isbn):
    return svd.predict(user_with_most_zeros, isbn).est

unread_books = books[~books["ISBN"].isin(
    ratings[ratings["User-ID"] == user_with_most_zeros]["ISBN"]
)].copy()

unread_books["Predicted-Rating"] = unread_books["ISBN"].map(predict_rating)

best_books = unread_books[unread_books["Predicted-Rating"] >= 8].copy()

"""### 4. Делаем предсказание LinReg для этих же книг."""

title = vectorizer.transform(best_books["Book-Title"])
author = author_encoder.transform(best_books["Book-Author"]).reshape(-1, 1)
publisher = publisher_encoder.transform(best_books["Publisher"]).reshape(-1, 1)
year = year_scaler.transform(best_books[["Year-Of-Publication"]])

X = hstack([title, author, publisher, year])

best_books["Predicted-Rating"] = linreg.predict(X)

"""### 5. Сортируем полученный на шаге 3 список по убыванию рейтинга линейной модели."""

best_books.sort_values(by="Predicted-Rating", ascending=False, inplace=True)

"""### 6. В конце файла комментарием записываем полученную рекомендацию."""

with open("best_books.txt", "w") as file:
    for index, row in best_books.iterrows():
        file.write(f"{row['Book-Title']} - Prediction: {row['Predicted-Rating']:.2f}\n")

with open("best_books.txt", "r") as best_books:
    for _ in range(5):
        print(best_books.readline().strip())

"""### Идея

То есть идея в том, чтобы сделать для пользователя индивидуальную рекомендацию, показывая в начале списка те книги, которые в целом могли бы иметь высокий рейтинг.

Обязательно сохраняйте готовую модель и добавляйте ее в свой репозиторий, потому что файл с сохраненной моделью используется в тестах.
"""