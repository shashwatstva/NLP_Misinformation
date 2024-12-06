import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset


## Predicting and evaluating the models for the original kaggle dataset
data = pd.read_csv('testdataset2.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
testdata = data.copy()

## Loading saved training data
X_train = pd.read_csv('trainingdataX.csv')['text']
y_train = pd.read_csv('trainingdataY.csv')['label']

print(f"X_train type: {type(X_train)}")
print(f"y_train type: {type(y_train)}")

## Using regular expressions to remove extra spaces, punctuations.

def preprocess_text(text):
    ## To remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    ## To remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

testdata['text'] = testdata['text'].apply(preprocess_text)

X_test = testdata['text']
y_test = testdata['label']

print(f"X_train type: {type(X_train)}")
print(f"X_test type: {type(X_test)}")

## Implementing TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(max_features=5000)

## Fit and transform on training data, transform test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape, y_train.shape)

baseline_model = LogisticRegression()
baseline_model.fit(X_train_tfidf, y_train)

## Predicting X_test values
y_pred_baseline = baseline_model.predict(X_test_tfidf)

predictions_baseline_df = pd.DataFrame({
    'text': X_test,
    'actual_label': y_test,
    'predicted_label': y_pred_baseline
})

## Saving predictions to CSV
## predictions_baseline_df.to_csv("2predictions_baseline.csv", index=False)

# Print model performance
print("Baseline Model Performance:")
print(classification_report(y_test, y_pred_baseline))

print('LR DONE')

#
# ### The entire code below has been commented out since its results are not included in the project report due to computational issues.
# ## Applying RoBERTa model on the entire dataset
#
# ## Loading the saved RoBERTa model and tokenizer
# model = RobertaForSequenceClassification.from_pretrained("./roberta_model")
# tokenizer = RobertaTokenizer.from_pretrained("./roberta_model")
#
# ## Test dataset for Hugging Face Trainer
# test_data = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))
#
# ## Defining function to tokenize the text before being used in the trainer
#
# def tokenize_text(statement):
#     return tokenizer(statement['text'], padding="max_length", truncation=True)
#
# test_data = test_data.map(tokenize_text, batched=True)
#
# ## Defining trainer with the loaded model
#
# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
# )
#
# ## Model Evaluation
#
# results = trainer.evaluate(eval_dataset= test_data)
# print("RoBERTa Model Performance:")
# print(results)
#
# ## Making predictions
#
# predictions = trainer.predict(test_data)
#
# ## Predicted class labels = y_pred_roberta
#
# y_pred_roberta = predictions.predictions.argmax(axis=-1)
#
# ## Storing predictions to a different file for further analysis.
#
# predictions_roberta_df = pd.DataFrame({
#     'text': X_test.values,
#     'actual_label': y_test.values,
#     'predicted_label': y_pred_roberta
# })
#
# predictions_roberta_df.to_csv("2predictions_roberta.csv", index=False)