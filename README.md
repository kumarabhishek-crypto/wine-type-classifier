# Wine Type Classification Website

This is a beginner-friendly Streamlit web app for the project **Wine Type Classification (Red/White) Using Regularized Logistic Regression**.

## Files you need

Keep these three files in the same folder:
- `app.py` — the website code
- `requirements.txt` — Python packages needed
- `wine_quality_merged (1)(1).csv` — your dataset

The CSV must include these columns:
`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, and `type`.

## Run it on your computer (optional)

1. Install Python if you do not already have it.
2. Open a terminal/command prompt in this folder.
3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   streamlit run app.py
   ```

5. Your browser should open the app automatically.

## Deploy online with Streamlit Community Cloud

1. Sign in to GitHub and create a **new public repository** (for example, `wine-type-classifier`).
2. Upload `app.py`, `requirements.txt`, and the CSV dataset into the repository's top level (not inside another folder).
3. Open Streamlit Community Cloud: https://share.streamlit.io/
4. Sign in with GitHub and choose **Create app**.
5. Select your repository, branch (usually `main`), and main file path `app.py`.
6. Click **Deploy** and wait for the build to finish.
7. Copy the public app URL and test the prediction form.

## Notes

- The app automatically trains and caches the model when it starts.
- It compares L1 and L2 regularization and tunes `C` using validation accuracy.
- It uses a held-out test split to calculate evaluation metrics.
- After evaluation, the deployed prediction model is refit on all labelled rows so it can use the available dataset for predictions.
- `quality` is deliberately excluded from input features; the model predicts `type` from the 11 physicochemical measurements.
- Do not put passwords, API keys, or private information in a public repository.
