# Fine-Tuning Llama 2 with QLoRA

This repository demonstrates the fine-tuning of the **Llama-2-7b-chat-hf** model using **QLoRA (Quantized LoRA)** for efficient and resource-friendly adaptation of large language models. The project focuses on training Llama 2 for instruction-following tasks using a custom dataset formatted to match the Llama 2 template.

---

## File Structure

This project consists of the following files:

1. **`app.py`**: The main application file that contains the code for fine-tuning the Llama 2 model.
2. **`requirements.txt`**: Lists the Python dependencies required to run the project.
3. **`README.md`**: Provides details about the project, its purpose, and how to use it.
4. **`.gitignore`** (Optional): A file to specify untracked files to ignore in the Git repository (e.g., logs, temporary files).

---

## 🚀 Overview

The project achieves:
- Fine-tuning the **Llama-2-7b-chat-hf** model using **4-bit precision**.
- Reducing GPU memory requirements via **QLoRA** and **parameter-efficient tuning**.
- Training the model on a dataset of 1,000 samples from the **mlabonne/guanaco-llama2-1k** dataset.
- Merging and pushing the fine-tuned model to the Hugging Face Hub for inference.

---

## 🛠️ Steps to Fine-Tune the Model

### **1. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **2. Configure Parameters**
The `app.py` file contains configurable parameters for:
- LoRA: Low-rank adaptation (e.g., rank 64, alpha 16, 4-bit quantization).
- Training: Batch size, learning rate, epochs, gradient accumulation, and precision settings.

### **3. Load and Preprocess the Dataset**
Use the preformatted dataset from Hugging Face:
- Train on 1,000 samples: [`mlabonne/guanaco-llama2-1k`](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
- Optionally, use the full dataset for more comprehensive fine-tuning.

### **4. Fine-Tune with QLoRA**
- Run the `app.py` script to load the **Llama-2-7b-chat-hf** model in 4-bit precision.
- Configure and apply **QLoRA** for memory-efficient tuning.
- The script handles the training pipeline and logs training progress.

### **5. Merge Weights and Save**
The script merges LoRA weights and saves the fine-tuned model locally or pushes it to the Hugging Face Hub.

### **6. Push to Hugging Face Hub**
Login to the Hugging Face Hub and push the fine-tuned model and tokenizer:
```bash
huggingface-cli login
```
```python
model.push_to_hub("your-username/Llama-2-7b-chat-finetune")
tokenizer.push_to_hub("your-username/Llama-2-7b-chat-finetune")
```

---

## 🧪 Example Inference
Use the fine-tuned model for text generation:
```python
from transformers import pipeline

pipe = pipeline(task="text-generation", model="your-username/Llama-2-7b-chat-finetune")
prompt = "What is a large language model?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

---

## 🖥️ Training Environment

- **Hardware**: Google Colab (15GB GPU, such as T4 or equivalent).
- **Software**: Python, PyTorch, and Hugging Face libraries.

---

## 📊 Training Configuration

- **Model**: `NousResearch/Llama-2-7b-chat-hf`
- **Dataset**: `mlabonne/guanaco-llama2-1k`
- **LoRA Parameters**:
  - Rank: 64
  - Alpha: 16
  - Dropout: 0.1
- **Training**:
  - Precision: 4-bit
  - Batch Size: 4
  - Epochs: 1
  - Learning Rate: 2e-4

---

## 🔑 Key Features

1. **QLoRA for Efficiency**: Enables fine-tuning of large models with minimal GPU resources.
2. **Llama 2 Prompt Template**: Ensures compatibility with Llama 2 chat models.
3. **Resource-Friendly**: Fits within the constraints of free-tier GPUs.

---

## 📊 Results and Usage
After fine-tuning, the model can be used for:
- Instruction following.
- Chat-based applications.
- General text generation tasks.

---

## 💑 Contributing

Feel free to fork this repository, raise issues, or submit pull requests to enhance the project.

---
