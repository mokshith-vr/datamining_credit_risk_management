# Credit Risk Management — CS F415 Data Mining
BITS Pilani Goa | Group C

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for credit risk prediction |
| `train.py` | Model training script |
| `analysis_model.ipynb` | Full analysis and model evaluation notebook |
| `loan_data.csv` | Dataset |
| `models/` | Pre-trained models (rf, xgb, lr, stacking) |

## Notes

- Pre-trained models are included in `models/` — no need to retrain
- To retrain from scratch: `python train.py`
