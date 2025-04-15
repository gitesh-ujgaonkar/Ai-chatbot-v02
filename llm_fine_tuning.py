# LLM Fine-Tuning with Conversational Personality
# This notebook demonstrates how to fine-tune a pre-trained language model
# to have a more friendly, conversational personality

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer

# Set up logging
logging.set_verbosity_info()
print("Setting up the environment...")

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU. This will be very slow!")

# 1. Load a pre-trained model
# We'll use a smaller model that can run in Colab, but you can scale up with more resources
print("\n1. Loading pre-trained model...")

# Configure quantization for efficient fine-tuning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the base model - using a smaller open-source model that can run in Colab
# For a production system, you'd want a larger model like Llama-2-7b or larger
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # A smaller model that can run in Colab
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Prepare the model for fine-tuning
print("\n2. Preparing model for fine-tuning...")
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
peft_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adjust based on model architecture
)

model = get_peft_model(model, peft_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# 3. Prepare training data
print("\n3. Loading and preparing training data...")

# For demonstration, we'll use a small subset of the Anthropic/HH-RLHF dataset
# which contains helpful and harmless conversational data
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")  # Limiting to 1000 examples for Colab

# Function to format conversations in the style the model expects
def format_conversation(example):
    # Create a friendly system prompt
    system_prompt = "You are a kind, helpful, and friendly AI assistant. You speak in a warm, conversational tone and genuinely care about helping people. You're thoughtful, empathetic, and aim to make people feel understood and supported."
    
    # Format the conversation
    conversation = f"<|system|>\n{system_prompt}\n<|user|>\n{example['chosen']}\n<|assistant|>\n"
    return {"text": conversation}

# Process the dataset
processed_dataset = dataset.map(format_conversation)
print(f"Processed {len(processed_dataset)} conversation examples")

# 4. Train the model
print("\n4. Starting fine-tuning process...")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # For demonstration; increase for better results
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none"  # Disable wandb reporting
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
    packing=True
)

print("Starting training... (This will take some time)")
trainer.train()
print("Training complete!")

# 5. Save the fine-tuned model
print("\n5. Saving the fine-tuned model...")
output_dir = "./friendly_assistant_model"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# 6. Test the model with a simple conversation
print("\n6. Testing the fine-tuned model...")

# Load the fine-tuned model
fine_tuned_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        trust_remote_code=True
    ),
    output_dir,
    torch_dtype=torch.float16
)

# Create a pipeline for text generation
generator = pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=tokenizer,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Test with a few examples
test_prompts = [
    "Hi there! How are you today?",
    "I'm feeling a bit sad today. Can you help me feel better?",
    "Can you explain quantum computing to me in simple terms?"
]

print("\nTesting the model with some example prompts:")
for prompt in test_prompts:
    system_prompt = "You are a kind, helpful, and friendly AI assistant. You speak in a warm, conversational tone and genuinely care about helping people."
    formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    response = generator(formatted_prompt)[0]['generated_text']
    assistant_response = response.split("<|assistant|>\n")[-1].strip()
    
    print(f"\nUser: {prompt}")
    print(f"Assistant: {assistant_response}")
    print("-" * 50)

# 7. Provide code to download the model
print("\n7. Code to download the fine-tuned model:")
print("""
# Run this code to download the model to your local machine
from google.colab import files

# Zip the model directory
!zip -r friendly_assistant_model.zip friendly_assistant_model

# Download the zip file
files.download('friendly_assistant_model.zip')
""")

print("\nImportant notes:")
print("1. This is a small demonstration model. For a truly capable assistant, you would need:")
print("   - A larger base model (7B+ parameters)")
print("   - More diverse and high-quality training data")
print("   - More training epochs and computational resources")
print("2. The model's capabilities are limited by the base model and training data")
print("3. For deployment, consider using a model serving platform or API")
