from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Define a held-out test example
instruction = "Identify the strongest positive feature, the primary problem, and one recommendation."
context = "Product: Smart Thermostat\nRating: 2/5\nReview: The screen is very bright and clear, but the Wi-Fi connection drops constantly every evening."

input_text = f"Instruction: {instruction}\nContext: {context}"

print("--- EVALUATION INPUT ---")
print(input_text)
print("\n" + "="*50 + "\n")

# 2. Test Pretrained Base Model
print("Loading Pretrained Base Model (google/flan-t5-small)...")
base_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

inputs = base_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
base_outputs = base_model.generate(**inputs, max_length=128)
base_response = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True)

print(f"\n[BASE MODEL OUTPUT]:\n{base_response}\n")

print("-" * 50)

# 3. Test Fine-Tuned Model
print("\nLoading Fine-Tuned Model from ./fine_tuned_flan_t5_support...")
ft_path = "./fine_tuned_flan_t5_support"
ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
ft_model = AutoModelForSeq2SeqLM.from_pretrained(ft_path)

ft_inputs = ft_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
ft_outputs = ft_model.generate(**ft_inputs, max_length=128)
ft_response = ft_tokenizer.decode(ft_outputs[0], skip_special_tokens=True)

print(f"\n[FINE-TUNED MODEL OUTPUT]:\n{ft_response}\n")
print("="*50)