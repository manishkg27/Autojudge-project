# AutoJudge: CP Problem Difficulty Estimator

## 📌 Project Overview

**AutoJudge** is a machine learning project designed to analyze competitive programming (CP) problems and predict their:

1.  **Problem Class** (Classification): Categorizes problems into _Easy_, _Medium_, or _Hard_.

    _Models are evaluated using accuracy and confusion matrices to analyze class-wise prediction performance across Easy, Medium, and Hard problems._

2.  **Problem Score** (Regression): Predicts the exact difficulty score (e.g., 800-3500 rating).

By analyzing the natural language in problem descriptions—specifically focusing on constraints, math notation (LaTeX), and algorithmic keywords—this model mimics how human competitive programmers assess difficulty.

## 📂 Dataset Used

The project utilizes a dataset of competitive programming problems (`problems_data.jsonl`) containing:

- **Description:** The main problem statement.
- **Input/Output:** Format specifications and constraints.
- **Tags/Class:** The ground truth difficulty level.
- **Score:** The numerical difficulty rating.

_Note: The raw data was cleaned to handle missing values, and missing scores were imputed based on class averages._

## ⚙️ Approach & Methodology

### 1. Data Preprocessing & Cleaning

Standard NLP cleaning is insufficient for CP problems because mathematical symbols are crucial.

- **Custom Regex Cleaning:** \* Converted LaTeX notation (e.g., `$10^9$`, `\le`) into standardized tokens (`10e9`, `<=`).
  - Preserved magnitude information (e.g., distinguishing between $N=100$ and $N=10^5$).
- **Normalization:** Lowercasing, stop-word removal, and tokenization.

### 2. Feature Engineering

We employed a **Hybrid Feature Approach** by stacking dense and sparse matrices:

- **TF-IDF Vectorization:** Extracted the top 3,000 frequent terms (n-grams) from the text.
- **Domain-Specific Features (Manual Extraction):**
  - **Keyword Counting:** Detected algorithmic terms like _'dp'_, _'graph'_, _'modulo'_, _'tree'_.
  - **Constraint Detection:** Binary flags for time limits, memory limits, and variable bounds ($10^5$ vs $10^{18}$).
  - **Text Statistics:** Length of description, input/output word counts.

### 3. Models Used

- **Classification:** Naive Bayes, Support Vector Machine (SVM), Random Forest, XGBoost.
- **Regression:** \* **Ensemble Methods:** Voting Regressor (combining XGBoost, Random Forest, and CatBoost).
  - **Gradient Boosting:** LightGBM, CatBoost, XGBoost.
  - **Linear:** Bayesian Ridge (to handle high-dimensional sparse data).

## 📊 Evaluation Metrics

| Metric       | Description                                                | Best Model Performance |
| :----------- | :--------------------------------------------------------- | :--------------------- |
| **Accuracy** | Percentage of correct Class predictions (Easy/Med/Hard)    | _~XX% (e.g., 85%)_     |
| **MAE**      | Mean Absolute Error (Average deviation in predicted score) | _~XX.X (e.g., 150.4)_  |
| **RMSE**     | Root Mean Squared Error (Penalizes large errors)           | _~XX.X_                |

_(Note: The Voting Regressor and CatBoost provided the lowest MAE in final testing.)_

## 🚀 Steps to Run Locally

### Prerequisites

- Python 3.8+
- pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/autojudge.git](https://github.com/your-username/autojudge.git)
    cd autojudge
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _(Ensure `lightgbm` and `catboost` are installed successfully. On Linux, you may need `libomp`)._

4.  **Run the Training Script:**

    ```bash
    python train_model.py
    ```

5.  **Launch the Web Interface:**
    ```bash
    streamlit run app.py
    ```

## 💻 Web Interface

The project includes a user-friendly web interface (built with Streamlit):

1.  **Input Area:** Paste any competitive programming problem description.
2.  **Constraint Extraction:** The app visualizes detected constraints (e.g., $N \le 10^5$).
3.  **Prediction:** Displays the predicted Difficulty Class and precise Score.

## 🎥 Demo Video

[Link to 2-3 Minute YouTube/Drive Demo Video]

## 👨‍💻 Author

**Manish** \* **System:** Arch Linux / Hyprland

- **Contact:** [Your Email or LinkedIn]
- **GitHub:** [Your GitHub Profile]

---

_Built with ❤️ using Python, Scikit-Learn, and Arch Linux._
