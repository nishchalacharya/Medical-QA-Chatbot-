# MedChat LLM: Fine-Tuned Llama Model for Medical Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange.svg)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/Unsloth-Latest-green.svg)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

MedChat LLM is a specialized medical chatbot built by fine-tuning Meta's Llama model on the MedQuad dataset. This project demonstrates advanced machine learning techniques for domain-specific language model adaptation, focusing on safe and helpful medical assistance. The implementation leverages cutting-edge tools like Unsloth for efficient fine-tuning and LoRA (Low-Rank Adaptation) for parameter-efficient training.

The project showcases expertise in:
- Large Language Model fine-tuning
- Medical NLP applications
- Efficient training methodologies (LoRA, quantization)
- Dataset preprocessing and formatting
- Inference optimization

## ✨ Features

- **Domain-Specific Fine-Tuning**: Specialized for medical Q&A using the MedQuad dataset
- **Efficient Training**: Utilizes LoRA and 4-bit quantization for memory-efficient fine-tuning
- **Safety-First Approach**: Includes system prompts for safe medical advice
- **Modular Architecture**: Clean separation of data processing, model training, and inference
- **Interactive Testing**: Jupyter notebook for easy model evaluation
- **Production-Ready**: Optimized for inference with proper tokenization and generation parameters

## 🏗️ Project Structure

### Core Components

#### `src/dataset.py`
Handles data loading and preprocessing for the MedQuad dataset:
- Loads the MedQuad dataset from Hugging Face Datasets
- Implements data cleaning filters (removes short/inadequate answers)
- Prepares raw medical Q&A pairs for fine-tuning

#### `src/formatting.py`
Converts raw data into chat-format suitable for instruction tuning:
- Structures data with system, user, and assistant roles
- Implements safety prompts (no diagnosis, consult doctor)
- Formats text for optimal model training

#### `src/model.py`
Manages model loading and LoRA configuration:
- Loads pre-trained TinyLlama model with Unsloth for efficiency
- Applies LoRA adapters to target attention and MLP layers
- Configures quantization (4-bit) for memory optimization
- Sets up gradient checkpointing for stable training

#### `src/train.py`
Orchestrates the complete fine-tuning pipeline:
- Integrates dataset loading, formatting, and model preparation
- Uses Supervised Fine-Tuning (SFT) with TRL library
- Implements training arguments optimized for medical domain
- Handles model saving and checkpointing

#### `src/inference.py`
Provides inference capabilities for the fine-tuned model:
- Loads trained model with LoRA weights
- Implements text generation with controlled parameters
- Handles tokenization and decoding for chat interactions

### Supporting Directories

#### `data/`
Reserved for local dataset storage and preprocessing artifacts.

#### `models/`
Contains saved model checkpoints and LoRA adapters post-training.

#### `notebooks/test.ipynb`
Interactive Jupyter notebook for:
- Model loading and testing
- Sample inference demonstrations
- Performance evaluation and debugging

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ VRAM for efficient fine-tuning

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd medchat_llm

# Install dependencies
pip install torch transformers datasets trl unsloth
```

## 📖 Usage

### Training the Model
```python
from src.train import main
main()
```
This will:
1. Load and preprocess the MedQuad dataset
2. Format data for chat-based fine-tuning
3. Load TinyLlama model with LoRA
4. Train for 200 steps with optimized hyperparameters
5. Save the fine-tuned model to `models/`

### Running Inference
```python
from src.inference import generate

prompt = """<|system|>
You are a helpful and safe medical assistant.
You do not provide diagnosis.
Always recommend consulting a doctor.

<|user|>
What are symptoms of diabetes?

<|assistant|>
"""

response = generate(prompt)
print(response)
```

### Testing in Notebook
Open `notebooks/test.ipynb` and run cells to interactively test the model.

## 🔧 Configuration

### Training Parameters
- **Model**: TinyLlama (1.1B parameters)
- **LoRA Rank**: 16
- **Batch Size**: 2 (with gradient accumulation of 4)
- **Max Steps**: 200
- **Learning Rate**: 2e-4
- **Quantization**: 4-bit

### Hyperparameters Explained

#### Model Configuration
- **max_seq_length**: 1024 - Maximum sequence length for input tokens. Limits context window to prevent excessive memory usage while covering most medical Q&A pairs.
- **load_in_4bit**: True - Enables 4-bit quantization, reducing model size by ~75% and memory footprint, allowing training on consumer GPUs with 8GB VRAM.

#### LoRA (Low-Rank Adaptation) Parameters
- **r (Rank)**: 16 - Dimensionality of the low-rank matrices. Higher values increase capacity but require more memory; 16 provides good balance for medical domain adaptation.
- **target_modules**: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] - Attention and MLP layers where LoRA adapters are applied. Covers all key transformer components for comprehensive fine-tuning.
- **lora_alpha**: 32 - Scaling factor for LoRA updates. Controls the magnitude of adapter contributions; higher values make fine-tuning more aggressive.
- **lora_dropout**: 0.05 - Dropout probability in LoRA layers. Prevents overfitting during training by randomly zeroing adapter outputs.
- **use_gradient_checkpointing**: True - Trades computation for memory by recomputing activations during backward pass, enabling training of larger models on limited VRAM.

#### Training Arguments
- **per_device_train_batch_size**: 2 - Number of samples processed per GPU per step. Small batch size due to memory constraints from 4-bit quantization.
- **gradient_accumulation_steps**: 4 - Accumulates gradients over 4 steps before updating weights, effectively creating a batch size of 8. Improves training stability and gradient quality.
- **max_steps**: 200 - Total training steps. Limited for demonstration; in production, train until convergence (typically 1-3 epochs depending on dataset size).
- **learning_rate**: 2e-4 - Step size for optimizer updates. Conservative rate prevents catastrophic forgetting while allowing adaptation to medical domain.
- **fp16**: True - Mixed precision training using 16-bit floats. Accelerates training and reduces memory usage without significant quality loss.
- **logging_steps**: 10 - Frequency of logging training metrics. Allows monitoring of loss, learning rate, and other metrics every 10 steps.
- **optim**: "adamw_8bit" - 8-bit AdamW optimizer. Memory-efficient variant that quantizes optimizer states, crucial for training large models on limited hardware.
- **seed**: 3407 - Random seed for reproducibility. Ensures consistent results across training runs for debugging and comparison.

#### Inference Parameters
- **max_new_tokens**: 200 - Maximum number of tokens to generate. Limits response length to prevent overly verbose or off-topic answers.
- **temperature**: 0.7 - Controls randomness in generation. Value between 0-1; 0.7 provides balanced creativity and coherence for medical responses.

### Safety Features
- System prompts enforce safe medical advice
- Filters out potentially harmful responses
- Recommends professional medical consultation

## 📊 Dataset

**MedQuad**: A comprehensive medical Q&A dataset containing:
- Medical questions and answers
- Curated from reliable sources
- Filtered for quality and relevance

## 🛠️ Technologies Used

- **Unsloth**: For fast, memory-efficient fine-tuning
- **Hugging Face Transformers**: Model and tokenizer management
- **TRL (Transformer Reinforcement Learning)**: Supervised fine-tuning
- **Datasets**: Data loading and processing
- **PyTorch**: Deep learning framework
- **LoRA**: Parameter-efficient fine-tuning technique

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Meta AI for the Llama model architecture
- Unsloth team for efficient fine-tuning tools
- Hugging Face for the MedQuad dataset and transformers library

## 📞 Contact

For questions or collaborations, please reach out via [your-email@example.com] or create an issue in the repository.

---

*This project demonstrates practical application of modern NLP techniques for healthcare AI, balancing technical excellence with ethical considerations in medical AI development.*