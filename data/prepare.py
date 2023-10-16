import pandas as pd
import re

#Prepare the Full Text Data
data_fulltext_dir = 'data/fullText/nike_data_fulltext.csv'

df_full = pd.read_csv(data_fulltext_dir)
data = df_full["Text"].str.cat(sep='\n')
text_data = re.sub(r'\n+', '\n', data).strip()  # Remove excess newline characters

with open("data/fullText/train_fulltext.txt", "w") as f:
    f.write(text_data)



#Prepare the Q and A Fine Tune Data
data_qanda_dir = 'data/QandA/nike_data_qanda.csv'

df_qanda = pd.read_csv('data/QandA/nike_data_qanda.csv')
data = df_qanda["Text"].str.cat(sep='\n')
qanda_data = re.sub(r'\n+', '\n', data).strip()  # Remove excess newline characters

with open("data/QandA/finetune_qanda.txt", "w") as f:
    f.write(qanda_data)

