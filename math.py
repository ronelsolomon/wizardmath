import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import sys

class LocalMathModelChat:
    """Local chat interface for specialized math models"""
    
    def __init__(self, model_name: str = "WizardMath", device: str = None):
        """
        Initialize the local math model chat interface
        
        Args:
            model_name: "WizardMath", "DeepSeek", or "ToRA"
            device: "cuda" (GPU), "cpu", or auto-detect
        """
        self.model_name = model_name
        
        # Model mappings
        self.models = {
            "WizardMath": "WizardLM/WizardMath-7B-V1.0",
            "DeepSeek": "deepseek-ai/deepseek-math-7b-instruct",
            "ToRA": "microsoft/ToRA-Code-7B-v0.1"
        }
        
        if model_name not in self.models:
            raise ValueError(f"Model must be one of: {list(self.models.keys())}")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"📦 Loading {model_name}...")
        print(f"🖥️  Using device: {self.device}")
        
        self.model_id = self.models[model_name]
        
        # Load tokenizer
        print("⏳ Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Load model with optimization
        print("⏳ Loading model (this may take 2-3 minutes on first run)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            load_in_8bit=True if self.device == "cuda" else False
        )
        
        self.model.eval()
        self.conversation_history = []
        print(f"✅ Successfully loaded {model_name}\n")
    
    def solve_math_problem(self, problem: str, show_steps: bool = True) -> str:
        """
        Solve a math problem with the model
        
        Args:
            problem: The math problem to solve
            show_steps: Whether to ask for step-by-step solution
        """
        if show_steps:
            prompt = f"""Solve this math problem step by step:

{problem}

Please show:
1. Your understanding of the problem
2. The approach/strategy
3. Step-by-step solution
4. Final answer with verification"""
        else:
            prompt = f"Solve: {problem}"
        
        return self._get_response(prompt)
    
    def explain_concept(self, concept: str) -> str:
        """Explain a mathematical concept"""
        prompt = f"""Explain the following mathematical concept clearly:

{concept}

Include:
1. Definition
2. Key properties
3. Real-world example
4. Common mistakes to avoid"""
        
        return self._get_response(prompt)
    
    def verify_solution(self, problem: str, solution: str) -> str:
        """Verify if a solution is correct"""
        prompt = f"""Check if this solution is correct:

Problem: {problem}

Proposed Solution: {solution}

Please verify and explain any errors if found."""
        
        return self._get_response(prompt)
    
    def _get_response(self, prompt: str, max_length: int = 512) -> str:
        """Get response from the local model"""
        try:
            # Store in history
            self.conversation_history.append({"role": "user", "content": prompt})
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            print("🤔 Generating response...\n")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Store response
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            return response_text
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def interactive_session(self):
        """Start an interactive chat session"""
        print(f"\n{'='*60}")
        print(f"Math Learning Chat with {self.model_name}")
        print(f"{'='*60}")
        print("Commands:")
        print("  'solve' - Solve a math problem")
        print("  'explain' - Explain a concept")
        print("  'verify' - Verify a solution")
        print("  'chat' - Free form question")
        print("  'quit' - Exit")
        print(f"{'='*60}\n")
        
        while True:
            cmd = input("Enter command: ").strip().lower()
            
            if cmd == "quit":
                print("Goodbye!")
                break
            
            elif cmd == "solve":
                problem = input("Enter the math problem: ").strip()
                if problem:
                    result = self.solve_math_problem(problem)
                    print(result)
                    print("\n" + "="*60 + "\n")
            
            elif cmd == "explain":
                concept = input("Enter the concept to explain: ").strip()
                if concept:
                    print("\n📚 Explaining...\n")
                    result = self.explain_concept(concept)
                    print(result)
                    print("\n" + "="*60 + "\n")
            
            elif cmd == "verify":
                problem = input("Enter the problem: ").strip()
                solution = input("Enter the proposed solution: ").strip()
                if problem and solution:
                    print("\n✓ Verifying...\n")
                    result = self.verify_solution(problem, solution)
                    print(result)
                    print("\n" + "="*60 + "\n")
            
            elif cmd == "chat":
                question = input("Enter your question: ").strip()
                if question:
                    result = self._get_response(question)
                    print(result)
                    print("\n" + "="*60 + "\n")
            
            else:
                print("Invalid command. Try again.\n")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize with your preferred model (WizardMath is most lightweight)
        chat = LocalMathModelChat(model_name="WizardMath")
        
        # Example: Solve a problem
        problem = "If a train travels 120 miles in 2 hours, what is its average speed?"
        print("\n📝 Example Problem:")
        print(f"Problem: {problem}\n")
        print("Solution:")
        result = chat.solve_math_problem(problem)
        print(result)
        
        # Uncomment to start interactive session
        print("\n" + "="*60)
        user_input = input("\nStart interactive session? (yes/no): ").strip().lower()
        if user_input == "yes":
            chat.interactive_session()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have installed required packages:")
        print("pip install torch transformers")