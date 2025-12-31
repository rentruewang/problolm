import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_perplexity(text):
    model_id = "gpt2" # Using GPT2 as a baseline "standard" probability model
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    if inputs['input_ids'].size(1) < 5: # Skip very short snippets
        return 999
        
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    
    return torch.exp(outputs.loss).item()

if __name__ == "__main__":
    files = sys.argv[1:]
    flagged = False
    
    for file_path in files:
        with open(file_path, 'r') as f:
            content = f.read()
            score = get_perplexity(content)
            print(f"File: {file_path} | Perplexity: {score:.2f}")
            
            # Threshold: Low perplexity = Likely AI
            if score < 12.0: 
                print(f"⚠️ WARNING: {file_path} looks highly predictable (Potential AI).")
                flagged = True
    
    if flagged:
        sys.exit(1)  # Exit with 1 to fail the build when AI code is detected