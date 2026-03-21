# Movie Rating Prediction: Binary Classification Challenge
**Author**: Jennifer Tsang

## Project Overview
To predict whether a client will rate a movie as "high" ($\ge 4$) or not. The problem is framed strictly as a binary classification task, focusing heavily on feature engineering and strict temporal validation. 

## Dataset
The data is sourced from a public Kaggle dataset [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset), specifically utilizing `rating.csv` and `movie.csv`. The massive scale of the data (over 10 million processed rows) necessitated memory-efficient data manipulation, iterative processing, and data downcasting techniques. Future iterations may incorporate `genome_scores.csv`, `genome_tags.csv`, `link.csv`, and `tag.csv` for deeper insights.

## Methodology
### 1. Data Structure & Modeling Approach
To prepare for modeling, the data was structured so that each instance (row) corresponds to a rating made by a specific user at a given point in time. The target response variable is binary:
* `1`: Rating $\ge 4$ (flag for "high" rating)
* `0`: Rating $< 4$ 

### 2. Feature Engineering & Data Leakage Prevention
The primary focus of this project was to implement a series of features with high predictive power while simulating an online predictions production setting. To strictly prevent data leakage, all historical features were calculated using chronologically sorted data, an expanding window, and a `shift(1)` operation. This ensures the model never uses future information to predict past events.

Key engineered features include:
* **User Historical Average**: The average rating a user has given prior to the current timestamp.
* **Movie Historical Average**: The average rating a movie has received from the public prior to the current timestamp.
* **User Genre Affinity**: A personalized, iterative metric calculating a user's historical average rating for specific genres (e.g., Action, Sci-Fi) to capture individualized taste without relying on standard collaborative filtering.

### 3. Machine Learning Models
Two machine learning models were trained to predict the response variable using the engineered predictive features.
* **Baseline Model**: Logistic Regression. Chosen for its simplicity, speed, and ability to establish a strong floor for linear relationships.
* **Main Model**: eXtreme Gradient Boosting (XGBoost). Selected for its robustness and ability to capture complex, non-linear interactions between users and genres.

To accurately assess the performance of the models and respect the chronological nature of the data, a `TimeSeriesSplit` cross-validation strategy was employed. 

### 4. Feature Importance
Feature importance was extracted to determine which engineered features had the highest impact on the model. **Gain** was selected as the importance metric because it measures the actual improvement in accuracy (reduction in loss) contributed by each feature, providing a more reliable business signal than simple weight/frequency counts.

## Technical Challenges Overcome
* **Scale and Memory Management**: Successfully processed a 10-million-row dataset without kernel crashes by implementing iterative genre processing, explicit garbage collection, and manual hyperparameter tuning (`subsample=0.6`, `colsample_bytree=0.6`) to limit memory overhead.

## Future Steps
* **Incorporate Genome Data**: Engineer features utilizing the `genome_scores.csv` and `genome_tags.csv` to capture a movie's underlying themes beyond basic genres.
* **Dynamic Time Weighting**: Analysis revealed a slight accuracy dip in the final time split, indicating Concept Drift. Future iterations will implement time-decay weighting to give higher importance to recent ratings, allowing the model to adapt as user preferences shift over time.
