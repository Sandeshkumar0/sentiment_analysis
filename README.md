# Sentiment Analysis (Streamlit)

This project is now a single-folder Streamlit app using the same sentiment logic as before:
- TF-IDF vectorizer (`vectorizer.pkl`)
- Logistic Regression model (`model.pkl`)
- Same preprocessing + confidence scoring

## Run

```powershell
cd C:\Users\KIIT0001\Desktop\sentiment-analysis\sentiment-final
C:\Users\KIIT0001\Desktop\sentiment-analysis\sentiment-final\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:
- http://localhost:8501

## Files

- `app.py` - Streamlit UI + prediction logic
- `model.pkl` - trained classifier
- `vectorizer.pkl` - TF-IDF vectorizer
- `train_model.py` - training script
- `sentiment_model.ipynb` - notebook
- `training_metrics.json` - saved metrics
