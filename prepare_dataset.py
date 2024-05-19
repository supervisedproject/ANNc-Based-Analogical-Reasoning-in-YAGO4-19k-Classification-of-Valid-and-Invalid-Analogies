# code 1:
pip install -q torch transformers pandas numpy tqdm matplotlib

# code 2:
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device)

# code 3:
import re
def clean_value(value):
    # Replace underscores with spaces for each variable
    value = value.replace('_', ' ')
    # Removes the wikipedia article ID from each variable
    value = re.sub(r' Q\d+$', '', value)
    return value
    
def clean_dataframe(df):
    # Apply the clean_value function to each element in the DataFrame
    cleaned_df = df.map(lambda x: clean_value(x) if isinstance(x, str) else x)
    #removes values with empty variables
    cleaned_df = cleaned_df.replace('', pd.NA).dropna()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df    

#code 4:
def get_embedding(sentences):
    #gets the embeddings, also applies mean pooling, truncation and padding.
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input.attention_mask
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    vector_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return vector_embeddings

# code 5:
import pandas as pd
def build_df(dataset_name,num_samples):
    #returns a dataframe with clean variables with a fix sample size.
    dataset_csv = pd.read_csv(dataset_name)
    dataset_csv = dataset_csv[:num_samples]
    dataset_csv = clean_dataframe(dataset_csv)
    return dataset_csv

# code 6:
#dataset_csv is the spreadsheet after being cleaned and with the selected number of samples
dataset_csv = build_df('shuffled_combined_dataset.csv',50000)

#code 7:
import numpy as np
from tqdm import tqdm
def build_dataset_with_emb(dataset_csv,batch_size = 1000):
    #compute embeddings for each of the rows and saves it in a column called word_embeddings in parallel (very fast)
    #be careful that this function is memory demmanding and boosting the variables too high can potentially crash your system
    Entity1 = dataset_csv['Entity1'].to_list()
    Entity2 = dataset_csv['Entity2'].to_list()
    Entity3 = dataset_csv['Entity3'].to_list()
    Entity4 = dataset_csv['Entity4'].to_list()

    x = []
    arange = np.arange(0, len(dataset_csv), batch_size, dtype=int)
    arange = np.append(arange, len(dataset_csv))
    #for idx,upper_limit in enumerate(arange[1:]):
    for idx, upper_limit in tqdm(enumerate(arange[1:]), total=len(arange[1:]), desc="Batches processed"):
        a = Entity1[arange[idx]:upper_limit]
        b = Entity2[arange[idx]:upper_limit]
        c = Entity3[arange[idx]:upper_limit]
        d = Entity4[arange[idx]:upper_limit]
        a = get_embedding(a)
        b = get_embedding(b)
        c = get_embedding(c)
        d = get_embedding(d)
        x.append(torch.stack((a,b,c,d)).permute(1,2,0))
    x = torch.cat(x, dim=0)
    df = pd.DataFrame({
        'word_embeddings': x.tolist()
    })
    #torch.tensor(df.iloc[0]['word_embeddings'])
    df = pd.concat([dataset_csv, df], axis=1)
    return df

# code 8:
#the dataset after we compute the embeddings and being added into the dataframe as a column
dataset_with_emb = build_dataset_with_emb(dataset_csv)

#code 9:
#save the dataset as a csv
dataset_with_emb.to_csv('50000_samples.csv', index=False)
