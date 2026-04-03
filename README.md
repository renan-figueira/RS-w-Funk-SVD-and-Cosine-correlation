# 🍿 Movie Recommender: Funk SVD + Cosine Similarity

A hybrid recommendation system built in Python. This project goes beyond simple rating correlation by combining **Machine Learning** (Matrix Factorization) with **Linear Algebra** (Cosine Similarity) to understand the true "essence" of a movie before recommending it.

### 🛠️ Tech Stack
* **Python 3**
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Surprise** (Funk SVD Implementation)
* **Scikit-Learn** (Cosine Similarity Calculation)

---

## 🧠 The Magic Behind the Code

Recommending movies merely by looking at who gave similar ratings can fail because people have different "baselines" (some are harsh critics, others give 5 stars to everything). To solve this, we split the problem into two stages:

1. **The Model (Funk SVD):** We use the algorithm made famous by the Netflix Prize. It analyzes the ratings matrix and extracts **100 latent factors** for each movie. The model autonomously "discovers" hidden features (such as action level, romance, or plot complexity), transforming each movie into a 100-dimensional mathematical vector.
2. **The Distance (Cosine):** With the movies transformed into vectors, we take our target movie (e.g., *Star Wars*) and calculate the Cosine Similarity against all others. The result? The system ignores user rating bias and recommends movies that share the exact same mathematical "DNA".

---

## 🚧 Development Notes (Gotchas!)

If you plan to run or explore this code, please note these two architecture decisions:

* **The `numpy < 2.0` trick:** The `scikit-surprise` library performs heavy calculations using Cython, which was optimized for the NumPy 1.x architecture. Since NumPy 2.0 introduced major under-the-hood structural changes, forcing a downgrade during the `pip install` prevents annoying import errors and ensures model stability.
* **Google Colab Execution:** This project was designed to run in the cloud. Because Colab has volatile storage (it deletes everything when the session disconnects), we use the `google.colab` library to mount Google Drive. This ensures data persistence without the need to manually upload files every time you run the script.

---

## 🚀 How to Run It Yourself

The data used is the **MovieLens 100k Dataset** ([GroupLens](https://grouplens.org/datasets/movielens/)).

### On Google Colab (Recommended)
1. Upload the `ml-100k` folder to your Google Drive.
2. Open the notebook, run the cells, and authorize access to your Drive when prompted.

### On your Local Machine
If you prefer to run this in VS Code or Jupyter Notebook, simply remove the Colab-specific lines and point the script to your local path:

```python
import pandas as pd

# Replace the Drive path with your local folder path
path = './ml-100k/'

# Load the data normally
df_ratings = pd.read_csv(f'{path}u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
```

## 🎯 Usage Example

Just call the function with the exact movie title and the number of recommendations you want:

```Python
# Finding recommendations for Star Wars
recommend_similar_movies_svd('Star Wars (1977)', top_n=5)
```
Expected Output:

```Plaintext
[3/3] Finding the 5 most similar movies to: 'Star Wars (1977)'

Empire Strikes Back, The (1980) (Similarity: 0.9231)
Return of the Jedi (1983) (Similarity: 0.8845)
Raiders of the Lost Ark (1981) (Similarity: 0.8120)
Stargate (1994) (Similarity: 0.7650)
Indiana Jones and the Last Crusade (1989) (Similarity: 0.7312)
