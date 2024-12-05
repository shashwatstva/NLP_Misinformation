import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

## Finetuning a RoBERTa model

data = pd.read_csv('preprocessed_data.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

## Splitting dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

print('preprocessing done')

print('starting BERT')
## Loading tokenizer and model for training.
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

## Training model on the training data
train_data = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
test_data = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))

## Defining function to tokenize the text before being used in the trainer
def tokenize_text(statement):

    return tokenizer(statement['text'], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_text, batched=True)
test_data = test_data.map(tokenize_text, batched=True)

## Training Arguments
training_args = TrainingArguments(
    output_dir="./results_roberta",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir='./logs_roberta',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
)


trainer.train()
## Saving the RoBERTa model and Tokenizer
model.save_pretrained("./roberta_model")
tokenizer.save_pretrained("./roberta_model")

print('model trained and saved')