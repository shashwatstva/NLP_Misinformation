import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

## Baseline Model: Using TF-IDF + Logistic Regression

data = pd.read_csv('preprocessed_data.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
df = data.copy()

X_train, X_test, y_train, y_test = train_test_split( data['text'], data['label'], test_size=0.2, random_state=42)

## Applying Tf- IDF

## Extract TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

## Applying Logistic Regression
baseline_model = LogisticRegression()
baseline_model.fit(X_train_tfidf, y_train)

## Predicting Values of the test set
y_pred_baseline = baseline_model.predict(X_test_tfidf)



predictions_baseline_df = pd.DataFrame({
    'text': X_test,
    'actual_label': y_test,
    'predicted_label': y_pred_baseline
})

## Storing predictions in another file for further use.
## Please note that label 1 refers to real news and 0 refers to fake news in the files.
predictions_baseline_df.to_csv("predictions_baseline.csv", index=False)

print("Baseline Model Performance:")
print(classification_report(y_test, y_pred_baseline))

print('LR DONE')
