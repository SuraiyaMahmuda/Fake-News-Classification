# Fake-News-Classification

## Introduction  
This project addresses the growing issue of fake news by leveraging Natural Language Processing (NLP) techniques and various machine learning classifiers such as **Logistic Regression**, **Multinomial Naive Bayes**, and **Bernoulli Naive Bayes**. Fake news has the potential to mislead people and cause serious harm. Identifying such misinformation automatically can help reduce its impact on society.

--- 

## Objective  
To build a classification model that can accurately differentiate between real and fake news articles using text-based features.

---

## Dataset  
- **Name:** `WELFake_Dataset.csv`  (Collected from Kaggle)
- **Dataset Link**: https://www.kaggle.com/datasets/studymart/welfake-dataset-for-fake-news
- **Source:** Includes labeled news articles classified as *Fake* or *True*  
- **Features Used:** Mainly the `text` content of the articles

--- 

## Process Overview  

### Data Preprocessing  
- Tokenization using `nltk`  
- Removing stopwords and punctuation  
- Lemmatization using `WordNetLemmatizer`  
- TF-IDF vectorization to convert text into numeric features  

### Exploratory Data Analysis (EDA)  
- Word clouds to visualize common terms  
- Class distribution analysis  
- Sample inspection using `pandas`   

### Model Building  
Several models were trained for comparison:
- Multinomial Naive Bayes (MNB)  
- Bernoulli Naive Bayes (BNB)  
- Logistic Regression (LR)   

### Evaluation Metrics  
- Accuracy  
- Confusion Matrix  
- Classification Report  
- ROC-AUC Score  
- ROC Curve using `RocCurveDisplay`  

--- 

## Results  
| Model                     | Accuracy |
|--------------------------|----------|
| Logistic Regression      | **95%** |
| Multinomial Naive Bayes  | 93%     |
| Bernoulli Naive Bayes    | 91%     |
- **Logistic Regression** showed the best performance with an accuracy of **~95%**.   
- Word clouds provided insight into commonly used words in both real and fake news.  
- The model successfully distinguishes between fake and real news articles.
  
--- 

## Tools and Libraries  
- `Pandas`, `NumPy` for data manipulation  
- `NLTK` for text preprocessing  
- `Scikit-learn` for modeling and evaluation  
- `Matplotlib`, `Seaborn`, `Plotly` for visualization  
- `WordCloud` for graphical insights

--- 

## Conclusion  
The project successfully demonstrated that NLP techniques combined with machine learning algorithms can be used to detect fake news with high accuracy. Such a system can be deployed in news aggregators, social media, or browser plugins to alert users about misinformation.

