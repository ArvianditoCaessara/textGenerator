from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from train import load_dataset, load_data_collator


# Configuration
finetune_file_path = "data/QandA/finetune_qanda.txt"
model_path = 'model/model_fulltext' #the pre-trained model
output_dir = 'model/model_finetuned_qanda' #fine-tuned model result
model_name = 'gpt2'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 50.0
save_steps = 50000

# Load the pre-trained model (after full-test training)
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

# Load the tokenizer
def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


# Fine Tune Function
def fine_tune(finetune_file_path,model_name,
              model_path,
              output_dir,
              overwrite_output_dir,
              per_device_train_batch_size,
              num_train_epochs,
              save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    #tokenizer = load_tokenizer(model_path)
    train_dataset = load_dataset(finetune_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
      
    model = load_model(model_path)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
                      output_dir=output_dir,
                      overwrite_output_dir=overwrite_output_dir,
                      per_device_train_batch_size=per_device_train_batch_size,
                      num_train_epochs=num_train_epochs,
                    )

    trainer = Trainer(
                  model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset,
              )
      
    trainer.train()
    trainer.save_model()



# Execute Fine-Tune
fine_tune(
    finetune_file_path=finetune_file_path,
    model_name=model_name,
    model_path=model_path, 
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)