
# 🎬 Movie Rating Prediction with Python

## 📌 Project Overview
This project builds a machine learning model to **predict IMDb movie ratings** based on features such as:
- 🎭 **Genre**
- 🎬 **Director**
- ⭐ **Actors**
- ⏳ **Duration**
- 📅 **Year of release**
- 🗳️ **Votes**

It uses regression techniques to estimate how users/critics would rate a movie.  

The notebook walks through:
- Data analysis & preprocessing  
- Feature engineering  
- Model training & evaluation  
- Saving the trained model  
- Making predictions for new movies  

---

## 📂 Repository Structure
```
├── data/
│   └── IMDb Movies India.csv       # Dataset
├── notebooks/
│   └── Movie_Rating_Prediction.ipynb   # Jupyter Notebook with step-by-step code
├── models/
│   └── imdb_rating_model.pkl       # Trained model (saved with joblib)
├── predict.py                      # Script to predict rating for new movies
└── README.md                       # Project documentation
```

---

## ⚙️ Installation
Clone this repository and install dependencies:

```bash
https://github.com/LNCDBBA21175/CODSOFT-Task-2.git
cd movie-rating-prediction
pip install -r requirements.txt
```

**Dependencies:**
- pandas  
- numpy  
- scikit-learn  
- joblib  
- matplotlib / seaborn (for visualization)  

---

## 🚀 Usage

### 1️⃣ Run Jupyter Notebook
Explore the notebook step by step:
```bash
jupyter notebook notebooks/Movie_Rating_Prediction.ipynb
```

### 2️⃣ Train and Save Model
The notebook trains models (Ridge, RandomForest, GradientBoosting) and saves the best one as:
```
models/imdb_rating_model.pkl
```

### 3️⃣ Predict Rating for New Movies
You can use the `predict.py` script to make predictions:

```bash
python predict.py
```

Example code inside `predict.py`:
```python
new_movie = pd.DataFrame([{
    "Name": "Test Movie",
    "Year": 2025,
    "Duration": 140,
    "Genre": "Drama, Action",
    "Director": "Rajkumar Hirani",
    "Actor 1": "Aamir Khan",
    "Actor 2": "Kareena Kapoor",
    "Actor 3": "Sharman Joshi",
    "Votes": "50000"
}])

predicted_rating = model.predict(X_new)[0]
print(f"Predicted IMDb Rating: {predicted_rating:.2f}")
```

---

## 📊 Model Performance
The notebook evaluates multiple models using:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

The best model is saved for future predictions.

---

## 🏆 Key Learnings
- Handling categorical & numerical features  
- One-hot encoding for genres, directors, actors  
- Applying regression models for prediction  
- Saving & loading models with `joblib`  
- Making predictions on unseen movies  

---

## ✨ Future Improvements
- Use deep learning (e.g., embeddings for text features)  
- Include more metadata (budget, language, country)  
- Deploy as a Flask/Django API or Streamlit app  
