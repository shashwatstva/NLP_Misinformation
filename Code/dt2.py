import pandas as pd
import re
from sklearn.model_selection import train_test_split

## Processing original dataset to evaluate the models

## Loading original Kaggle dataset
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

## Data used for the primary research
true_subset = pd.read_csv("True_data.csv")
fake_subset = pd.read_csv("Fake_data.csv")

## Assigning labels: 0 for fake news and 1 for real news

true_df['label'] = 1
fake_df['label'] = 0


true_subset['label'] = 1  # 1 for real news
fake_subset['label'] = 0  # 0 for fake news

## Combining both the dataframes to create a master set of news data

data = pd.concat([true_df, fake_df], ignore_index=True)
subset = pd.concat([true_subset, fake_subset], ignore_index=True)

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    subset['text'], subset['label'], test_size=0.2, random_state=42)

testdata = data[~data['text'].isin(X_train.values)]

## The below 3 lines (save as csv) are commented out. However, they were run and the mentioned sets were saved as csv for further analysis
# testdata.to_csv('testdataset2.csv', index=False)
# X_train.to_csv('trainingdataX.csv', index=False)
# y_train.to_csv('trainingdataY.csv', index=False)

## Using regular expressions to remove extra spaces, punctuations.

def preprocess_text(text):
    ## To remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    ## To remove Punctuations
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

testdata['text'] = testdata['text'].apply(preprocess_text)

X_test = testdata[['text']]
y_test = testdata[['label']]

print('done')
