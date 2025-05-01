# Llama2_Fine_Tuning

# 🦙 Fine-Tuning-LLaMA

## 🧪 Practical Introduction to LLaMA 2 Fine-Tuning

In this project, I worked on fine-tuning the **LLaMA 2** model using **QLoRA** on a **T4 GPU** via Google Colab. This demonstrates how to fine-tune a large language model with limited VRAM using efficient parameter-efficient fine-tuning techniques.

---

## 📌 Overview

Fine-tuning a 7B parameter LLaMA 2 model is challenging due to VRAM constraints. This project explores **QLoRA**, which enables 4-bit fine-tuning while preserving model quality. It uses the Hugging Face ecosystem, incorporating:

- `transformers`
- `accelerate`
- `peft`
- `trl`
- `bitsandbytes`

---

## ⚙️ Prerequisites

- Google Colab or a local setup with an **NVIDIA GPU (T4 or better)**
- Python **3.8+**

---

## 🔧 Configuration and Setup

### 📚 Library Overview

- **Transformers**: Load and run pre-trained models and tokenizers.
- **Accelerate**: Simplifies multi-GPU setup and training.
- **PEFT**: Enables efficient fine-tuning with LoRA.
- **TRL**: Offers trainer utilities for RLHF-style fine-tuning.
- **BitsAndBytes**: Allows low-level memory optimizations (4-bit quantization).


## Preparing the Dataset

Load the dataset mlabonne/guanaco-llama2-1k using Hugging Face's datasets library:

```python
from datasets import load_dataset

dataset = load_dataset('mlabonne/guanaco-llama2-1k', split='train')
```

## Model Initialization

Initialize the Llama 2 model for fine-tuning with 4-bit precision adjustments:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('NousResearch/Llama-2-7b-chat-hf', use_4bit=True)
```

## Training Configuration 

Configure the training parameters such as batch size, number of epochs, and learning rate:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-5)
```

## Usage
After training, use the model to generate text or for other NLP tasks:
```python

from transformers import pipeline

generator = pipeline('text-generation', model='path_to_fine_tuned_model')
print(generator("Prompt for the model", max_length=50))
```
