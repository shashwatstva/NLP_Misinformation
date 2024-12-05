import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset


data = pd.read_csv('preprocessed_data.csv')
# Drop columns with "Unnamed" in their name
df= data.copy()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

print('preprocessing done')

## Loading the saved RoBERTa model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("./roberta_model")
tokenizer = RobertaTokenizer.from_pretrained("./roberta_model")

## Test dataset for Hugging Face Trainer
test_data = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))

## Defining function to tokenize the text before being used in the trainer

def tokenize_text(statement):
    return tokenizer(statement['text'], padding="max_length", truncation=True)

test_data = test_data.map(tokenize_text, batched=True)

## Defining trainer with the loaded model
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
)

## Model Evaluation
results = trainer.evaluate(eval_dataset= test_data)
print("RoBERTa Model Performance:")
print(results)

## Making predictions
predictions = trainer.predict(test_data)

## Predicted class labels = y_pred_roberta
y_pred_roberta = predictions.predictions.argmax(axis=-1)

## Storing predictions to a different file for further analysis.
predictions_roberta_df = pd.DataFrame({
    'text': X_test.values,
    'actual_label': y_test.values,
    'predicted_label': y_pred_roberta
})

predictions_roberta_df.to_csv("predictions_roberta.csv", index=False)