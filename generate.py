from transformers import GPT2Tokenizer
from fine_tune import load_model

# Config
#model_path = 'model/model_fulltext' #The model that only trained with Full-Text
model_path = 'model/model_finetuned_qanda' #The model after Q and A fine-tuned
question = "[Q] When was the company founded?"
max_len = 50


# Load the tokenizer
def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# Generate the text
def generate_text(model_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))


generate_text(model_path, question, max_len) 