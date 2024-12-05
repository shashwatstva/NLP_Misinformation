import pandas as pd
import re

## Data preprocessing.

true_df = pd.read_csv("True_data.csv")
fake_df = pd.read_csv("Fake_data.csv")

## Assigning labels: 0 for fake news and 1 for real news

true_df['label'] = 1
fake_df['label'] = 0

## Combining both the dataframes to create a master set of news data
data = pd.concat([true_df, fake_df], ignore_index=True)

## Using regular expressions to remove extra spaces, punctuations.
def preprocess_text(text):
    ## To remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    ## To remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

data['text'] = data['text'].apply(preprocess_text)
df= data.copy()
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


data.to_csv('preprocessed_data.csv', index= False)


print('preprocessing done')