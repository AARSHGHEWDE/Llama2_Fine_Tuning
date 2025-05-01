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
