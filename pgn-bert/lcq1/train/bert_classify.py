from datasets import load_dataset
from datasets import DatasetDict
import sys,json
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_metric
from transformers import AutoModelForSequenceClassification
import numpy as np

fold = str(sys.argv[1])

dataset_ = load_dataset('json',data_files='bert_rankshuf%s.json'%(fold))['train']
dataset_ = dataset_.train_test_split(test_size=0.1)

tvd = dataset_['test'].train_test_split(test_size=0.5)

dataset = DatasetDict({
    'train': dataset_['train'],
    'test': tvd['train'],
    'valid': tvd['test']})

print(dataset)
print(dataset['train'][0])
print(dataset['test'][0])
print(dataset['valid'][0])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_imdb = dataset.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)



def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./bert_classify_results_%s"%(fold),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    evaluation_strategy="steps",
    eval_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()
