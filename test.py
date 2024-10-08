import warnings
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")

from transformers import BertTokenizer, BertForNextSentencePrediction, GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import time
import numpy as np
import psutil
import os
import json

# Load the BERT tokenizer and model for NSP (from saved results)
bert_tokenizer = BertTokenizer.from_pretrained('./results/bert')
bert_model = BertForNextSentencePrediction.from_pretrained('./results/bert')

# Load the GPT-2 tokenizer and model for next sentence generation (from saved results)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('./results/gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('./results/gpt2')

# Training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    warmup_steps=100,
    max_grad_norm=1.0,
    adam_epsilon=1e-8,
)

def predict_nsp(model, tokenizer, sentence_a, sentence_b):
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    
    follows = probs[0] > probs[1]
    return follows, probs[0].item()

def generate_next_sentence(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    start_time = time.time()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    inference_time = time.time() - start_time
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    next_sentence = generated_text[len(input_text):].strip()
    
    return next_sentence, inference_time

def compute_metrics(sentence_pairs, expected_results):
    y_true = expected_results
    y_pred = []
    y_scores = []
    inference_times = []
    
    for (sentence_a, sentence_b), expected in zip(sentence_pairs, expected_results):
        start_time = time.time()
        follows, prob = predict_nsp(bert_model, bert_tokenizer, sentence_a, sentence_b)
        inference_time = time.time() - start_time
        
        y_pred.append(int(follows))
        y_scores.append(prob)
        inference_times.append(inference_time)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc_roc = roc_auc_score(y_true, y_scores)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'avg_inference_time': np.mean(inference_times),
        'throughput': 1 / np.mean(inference_times),
        'y_true': y_true,
        'y_scores': y_scores
    }

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def calculate_gpt2_perplexity(model, tokenizer, sentence_pairs):
    model.eval()
    total_loss = 0
    total_length = 0

    with torch.no_grad():
        for sentence_a, sentence_b in sentence_pairs:
            input_text = f"{sentence_a} {sentence_b}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_length += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

if __name__ == "__main__":
    sentence_pairs = [
        ("He is a talented musician.", "He can play multiple instruments."),
        ("The sun is shining.", "It's a beautiful day outside."),
        ("I love reading books.", "My favorite genre is science fiction."),
        ("The cat is sleeping.", "The dog is barking loudly."),
        ("She went to the grocery store.", "She bought milk and bread."),
        ("The car won't start.", "The battery is dead."),
        ("I'm learning to code.", "I hate programming and computers."),
        ("The movie was exciting.", "I fell asleep halfway through."),
    ]
    expected_results = [1, 1, 1, 0, 1, 1, 0, 0]

    # Compute metrics
    metrics = compute_metrics(sentence_pairs, expected_results)

    # Get model sizes
    bert_size = get_model_size(bert_model)
    gpt2_size = get_model_size(gpt2_model)

    # Get memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024**2  # in MB

    # Calculate GPT-2 metrics
    gpt2_perplexity = calculate_gpt2_perplexity(gpt2_model, gpt2_tokenizer, sentence_pairs)
    gpt2_inference_times = []
    for sentence_a, _ in sentence_pairs:
        _, gen_time = generate_next_sentence(gpt2_model, gpt2_tokenizer, sentence_a)
        gpt2_inference_times.append(gen_time)

    # Prepare metrics data for saving
    metrics_data = {
        "performance_metrics": {
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
            "auc_roc": metrics['auc_roc'],
            "error_rate": 1 - metrics['accuracy']  # Add this line
        },
        "inference_metrics": {
            "avg_inference_time": metrics['avg_inference_time'],
            "throughput": metrics['throughput']
        },
        "resource_utilization": {
            "bert_model_size": bert_size,
            "gpt2_model_size": gpt2_size,
            "memory_usage": memory_usage
        },
        "gpt2_metrics": {
            "perplexity": gpt2_perplexity,
            "avg_inference_time": np.mean(gpt2_inference_times),
            "throughput": 1 / np.mean(gpt2_inference_times)
        }
    }

    # Save metrics data to JSON file
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print("Metrics and parameters have been saved to 'model_metrics.json'")

    # Generate and print sentences for each input
    print("\nGenerated sentences for predefined inputs:")
    for sentence_a, _ in sentence_pairs:
        generated_next, gen_time = generate_next_sentence(gpt2_model, gpt2_tokenizer, sentence_a)
        print(f"Input: {sentence_a}")
        print(f"Generated next sentence: {generated_next}")
        print(f"Generation time: {gen_time:.5f} seconds")
        print()