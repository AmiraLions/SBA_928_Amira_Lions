"""
Model Trainer for SBA 928: Fine-Tuning google/flan-t5-small
Author: Amira Lions
Description: Loads the processed customer support dataset, tokenizes text, 
             and fine-tunes the google/flan-t5-small model using Hugging Face Trainer.
"""

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from pathlib import Path

def train_model():
    # 1. Load the processed dataset created by dataset_builder.py
    dataset_path = Path("processed_support_dataset.csv")
    if not dataset_path.exists():
        print(f"Error: Could not find {dataset_path}. Please run dataset_builder.py first.")
        return

    print("Loading processed dataset...")
    df = pd.read_csv(dataset_path)
    
    # For efficient fine-tuning during the assessment, we sample a subset of 500 rows
    df_subset = df.head(500)
    print(f"Using {len(df_subset)} examples for fine-tuning.")

    # Convert pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df_subset)
    
    # Split into train and validation sets (90% train, 10% eval)
    split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Training rows: {len(split_dataset['train'])}, Validation rows: {len(split_dataset['test'])}")

    # 2. Initialize Model and Tokenizer
    model_checkpoint = "google/flan-t5-small"
    print(f"Loading tokenizer and model for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # 3. Tokenization Function
    def preprocess_function(examples):
        inputs = [
            f"Instruction: {instr}\nContext: {ctx}" 
            for instr, ctx in zip(examples["instruction"], examples["context"])
        ]
        targets = examples["target"]

        model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_train = split_dataset['train'].map(preprocess_function, batched=True)
    tokenized_eval = split_dataset['test'].map(preprocess_function, batched=True)

    # 4. Configure Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=1,
        predict_with_generate=True,
        logging_steps=10,
        save_strategy="epoch"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    # 6. Execute Fine-Tuning
    print("Starting model fine-tuning...")
    trainer.train()

    # Save the fine-tuned model and tokenizer
    output_dir = Path("./fine_tuned_flan_t5_support")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model successfully trained and saved to {output_dir}")

if __name__ == "__main__":
    train_model()