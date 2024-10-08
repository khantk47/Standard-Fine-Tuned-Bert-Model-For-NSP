import warnings
from transformers import BertTokenizer, BertForNextSentencePrediction, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time
import psutil
import os

# Add this line to filter out the specific warning
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")

# Load BERT tokenizer and model for NSP
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# Load GPT-2 tokenizer and model for sentence generation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the WikiText dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_texts = dataset['train']['text']
val_texts = dataset['validation']['text']

# Modify the prepare_sentence_pairs function to return a Dataset object
def prepare_sentence_pairs(texts, max_length=32):
    """ Prepares pairs of sentences from the WikiText dataset for NSP training. """
    sentence_pairs = []
    labels = []

    for text in texts:
        sentences = text.split(". ")
        for i in range(len(sentences) - 1):
            sentence_a = sentences[i]
            sentence_b = sentences[i + 1]

            # Add the pair to the dataset with label 0 (i.e., B follows A)
            if sentence_a and sentence_b:
                encoded = bert_tokenizer(sentence_a, sentence_b,
                                    return_tensors='pt',
                                    padding='max_length',
                                    truncation=True,
                                    max_length=max_length)
                encoded['labels'] = torch.tensor([0])  # Label 0 means B follows A
                sentence_pairs.append({key: value.squeeze() for key, value in encoded.items()})

            # Add a non-sequential pair to create negative examples (label 1)
            if i + 2 < len(sentences):
                sentence_b = sentences[i + 2]
                if sentence_a and sentence_b:
                    encoded = bert_tokenizer(sentence_a, sentence_b,
                                        return_tensors='pt',
                                        padding='max_length',
                                        truncation=True,
                                        max_length=max_length)
                    encoded['labels'] = torch.tensor([1])  # Label 1 means B does not follow A
                    sentence_pairs.append({key: value.squeeze() for key, value in encoded.items()})

    return Dataset.from_list(sentence_pairs)

# Prepare datasets for training and validation
train_data = train_texts[:500]
val_data = val_texts[:100]
train_dataset = prepare_sentence_pairs(train_data)
val_dataset = prepare_sentence_pairs(val_data)

# Define a custom data collator
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# Define a custom evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        "accuracy": accuracy,
        "f1": f1,
    }

# Modify Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Define Trainer
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and evaluate the model
if __name__ == "__main__":
    start_time = time.time()
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # in MB
    
    # Train the model
    train_result = trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # in MB
    memory_used = final_memory - initial_memory
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    
    # Print results
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Final evaluation results: {eval_results}")
    print(f"Training loss: {train_result.training_loss}")
    
    # Save the trained models
    bert_model.save_pretrained('./results/bert')
    bert_tokenizer.save_pretrained('./results/bert')
    gpt2_model.save_pretrained('./results/gpt2')
    gpt2_tokenizer.save_pretrained('./results/gpt2')