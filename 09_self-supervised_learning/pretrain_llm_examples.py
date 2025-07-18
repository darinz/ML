"""
Pretrained Large Language Models: Educational Python Examples

This file demonstrates the main concepts from 02_pretrain_llm.md:
1. Language modeling and chain rule
2. Transformer input/output interface
3. Autoregressive text generation (sampling, temperature)
4. Finetuning (conceptual, with HuggingFace)
5. Zero-shot and in-context learning (prompting)

All code is for educational purposes and uses synthetic or small real data.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline, Trainer, TrainingArguments
)
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# 1. LANGUAGE MODELING AND CHAIN RULE
# ============================================================================

def chain_rule_example():
    """
    Demonstrate the chain rule for language modeling with a toy example.
    The chain rule decomposes the joint probability of a sequence into a product of conditionals:
        P(x1, x2, ..., xT) = P(x1) * P(x2|x1) * ... * P(xT|x1,...,xT-1)
    """
    print("\n=== LANGUAGE MODELING: CHAIN RULE EXAMPLE ===")
    # Suppose we have a vocabulary of 3 words: ["I", "like", "cats"]
    vocab = ["I", "like", "cats"]
    V = len(vocab)
    # A toy sentence: "I like cats"
    x = [0, 1, 2]  # indices in vocab
    T = len(x)
    # Suppose we have a model that gives the following conditional probabilities:
    p = np.array([
        [0.7, 0.2, 0.1],  # p(x1)
        [0.1, 0.8, 0.1],  # p(x2|x1)
        [0.2, 0.2, 0.6],  # p(x3|x1,x2)
    ])
    # Compute joint probability using chain rule
    joint = p[0, x[0]] * p[1, x[1]] * p[2, x[2]]
    print(f"Sentence: {' '.join([vocab[i] for i in x])}")
    print(f"Joint probability (chain rule): {joint:.4f}")
    print("\nExplanation:")
    print("- The model predicts the probability of each word given the previous context.")
    print("- The joint probability is the product of these conditionals.")
    print("- This is the foundation of autoregressive language modeling.")

# ============================================================================
# 2. TRANSFORMER INPUT/OUTPUT INTERFACE
# ============================================================================

def transformer_io_example():
    """
    Show how to tokenize text and get logits from a pretrained Transformer.
    Demonstrates the input/output interface of modern LLMs.
    """
    print("\n=== TRANSFORMER INPUT/OUTPUT INTERFACE ===")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    text = "The speed of light is"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    print(f"Input text: {text}")
    print(f"Tokenized input IDs: {inputs['input_ids'][0].tolist()}")
    print(f"Logits shape: {logits.shape} (batch, seq_len, vocab_size)")
    # Show the next-token logits
    next_token_logits = logits[0, -1]
    probs = torch.softmax(next_token_logits, dim=-1)
    topk = torch.topk(probs, 5)
    print("Top 5 next-token predictions:")
    for idx, prob in zip(topk.indices, topk.values):
        print(f"  {tokenizer.decode(idx.item())!r}: {prob.item():.3f}")
    print("\nExplanation:")
    print("- The model outputs a probability distribution over the vocabulary for the next token.")
    print("- The highest-probability tokens are likely continuations.")

# ============================================================================
# 3. AUTOREGRESSIVE TEXT GENERATION (SAMPLING, TEMPERATURE)
# ============================================================================

def autoregressive_generation_example():
    """
    Demonstrate text generation with temperature sampling.
    Shows how temperature affects randomness and creativity in generation.
    """
    print("\n=== AUTOREGRESSIVE TEXT GENERATION (TEMPERATURE) ===")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    prompt = "Q: 2 ~ 3 = ? A: 5 Q: 6 ~ 7 = ? A: 13 Q: 15 ~ 2 = ? A:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Generate with different temperatures
    for temp in [1.0, 0.7, 0.2]:
        print(f"\nTemperature: {temp}")
        output = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=temp,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Generated: {generated.strip()}")
    print("\nExplanation:")
    print("- Higher temperature increases randomness and diversity.")
    print("- Lower temperature makes output more deterministic and focused.")

# ============================================================================
# 4. FINETUNING (CONCEPTUAL, WITH HUGGINGFACE)
# ============================================================================

def finetuning_example():
    """
    Conceptual example of finetuning a language model using HuggingFace Trainer.
    (This is a template; actual finetuning requires a dataset and compute.)
    """
    print("\n=== FINETUNING EXAMPLE (CONCEPTUAL) ===")
    print("Suppose we have a dataset of (input, label) pairs for a classification task.")
    print("We add a linear head on top of the pretrained model and train both the head and the model.")
    print("With HuggingFace, you would use Trainer and TrainingArguments, e.g.:")
    print("""
from transformers import Trainer, TrainingArguments
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./results', num_train_epochs=3),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
    """)
    print("This will finetune the model for your specific task.")
    print("\nBest practices:")
    print("- Use a small learning rate for the pretrained layers.")
    print("- Use a larger learning rate for the new head.")
    print("- Monitor for overfitting and catastrophic forgetting.")

# ============================================================================
# 5. ZERO-SHOT AND IN-CONTEXT LEARNING (PROMPTING)
# ============================================================================

def zero_shot_and_in_context_example():
    """
    Show zero-shot and in-context learning with prompting.
    Demonstrates how LLMs can adapt to new tasks without parameter updates.
    """
    print("\n=== ZERO-SHOT AND IN-CONTEXT LEARNING ===")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # Zero-shot: ask a question with no examples
    prompt_zero_shot = "Is the speed of light a universal constant?"
    input_ids = tokenizer(prompt_zero_shot, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Zero-shot prompt: {prompt_zero_shot}")
    print(f"Model answer: {answer.strip()}")
    # In-context: give a few examples, then a new question
    prompt_few_shot = (
        "Q: 2 ~ 3 = ? A: 5\n"
        "Q: 6 ~ 7 = ? A: 13\n"
        "Q: 15 ~ 2 = ? A:"
    )
    input_ids = tokenizer(prompt_few_shot, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=5, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nIn-context prompt:\n{prompt_few_shot}")
    print(f"Model answer: {answer.strip()}")
    print("\nExplanation:")
    print("- Zero-shot: The model uses pretraining knowledge to answer new questions.")
    print("- In-context: The model infers the pattern from examples in the prompt.")
    print("- This is the basis for prompt engineering and few-shot learning.")

# ============================================================================
# PRACTICAL NOTES AND BEST PRACTICES
# ============================================================================

def practical_notes():
    """
    Print practical notes and best practices for LLM pretraining and adaptation.
    """
    print("\n=== PRACTICAL NOTES AND BEST PRACTICES ===")
    print("- Pretraining requires massive data and compute; use existing models when possible.")
    print("- Always preprocess and clean your data.")
    print("- Use prompt engineering to adapt LLMs to new tasks.")
    print("- Monitor for bias and hallucination in outputs.")
    print("- Evaluate on real downstream tasks, not just perplexity.")
    print("- Use finetuning for domain adaptation when you have labeled data.")
    print("- For most users, in-context learning and prompting are the most practical ways to use LLMs.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    chain_rule_example()
    transformer_io_example()
    autoregressive_generation_example()
    finetuning_example()
    zero_shot_and_in_context_example()
    practical_notes()
    print("\nAll examples completed. See comments for further explanation and best practices.")

if __name__ == "__main__":
    main() 