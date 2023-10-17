from transformers import TextDataset, DataCollatorForLanguageModeling, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Configuration
train_file_path = "data/fullText/train_fulltext.txt"
model_name = 'gpt2'
output_dir = 'model/model_fulltext'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 50.0
save_steps = 50000

# Load Dataset
def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


# Data Collator
def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator


# Train Function
def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
      
    model = GPT2LMHeadModel.from_pretrained(model_name)

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
    return trainer


# Execute Training
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)