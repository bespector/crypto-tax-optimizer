# Crypto Tax Lot Analyzer

Use this app to match your crypto currency sells with a previous buy trade based off an accounting method. For more information see the Introduction and Optimization Setup Pages in the Streamlit app:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://34.236.158.237:8501/)


We currently support the following accounting methods:
1. FIFO (First In First Out)
2. HIFO (Highest In First Out)
3. LIFO (Last In First Out)
4. TAX_OPTIMAL (This is defined as the solution that minimizes tax liability using long and short term tax rates)

## Run this demo locally
```
pip install --upgrade streamlit
streamlit run app.py
```

![alt text](docs/Streamlit_Image.png?raw=true "Title")
