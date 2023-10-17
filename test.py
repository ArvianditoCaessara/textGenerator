from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

#Config
model1_path = 'model/model_fulltext'
model2_path = 'model/model_finetuned_qanda'
tokenizer_path = 'model/model_fulltext'
sequence1 = "[Q] When was the company founded?"
max_len = 50



# Functions
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model_path, tokenizer_path, sequence, max_length):
    
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
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



generate_text(model1_path, tokenizer_path, sequence1, max_len) 