# WizardMath Chat Interface

A Python interface for interacting with state-of-the-art mathematical language models including WizardMath, DeepSeek, and ToRA.

## 🌟 Features

- **Multiple Models**: Choose between different math-focused models
- **Local Inference**: Run models locally with GPU acceleration
- **Interactive Chat**: Natural language interface for math problem solving
- **Easy Setup**: Simple installation and usage

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA for GPU acceleration)
- Hugging Face Transformers

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wizardmath.git
cd wizardmath
```

2. Install dependencies:
```bash
pip install torch transformers accelerate
```

### Usage

Run the interactive chat interface:
```bash
python math.py
```

Select your preferred model when prompted:
- WizardMath
- DeepSeek
- ToRA

## 🤖 Available Models

### WizardMath
- **Model**: WizardMath-7B-V1.0
- **Strengths**: General math problem solving, step-by-step reasoning
- **Best for**: Complex mathematical proofs and explanations

### DeepSeek
- **Model**: deepseek-math-7b-instruct
- **Strengths**: Mathematical reasoning and computation
- **Best for**: Precise calculations and formula derivations

### ToRA
- **Model**: ToRA-Code-7B-v0.1
- **Strengths**: Combining natural language with code-based solutions
- **Best for**: Problems requiring computational solutions

## 💡 Example Queries

- "Solve for x: 2x + 5 = 15"
- "Explain the Pythagorean theorem"
- "What is the integral of x^2?"
- "Prove that the square root of 2 is irrational"

## 📊 Performance Tips

- For best performance, use a CUDA-enabled GPU
- Reduce response length for faster answers
- The first run will download the model (several GB)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WizardLM](https://huggingface.co/WizardLM)
- [DeepSeek](https://huggingface.co/deepseek-ai)
- [Microsoft ToRA](https://huggingface.co/microsoft/ToRA-Code-7B-v0.1)
