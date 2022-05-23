import json
import random
import copy
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
from model import BertSimilarity
random.seed(1234)





@torch.no_grad()
def bert_embedding(logs_file: str):
    logs = load_logs(logs_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    d = {}
    for index, raw in tqdm(logs.iterrows()):
        # print(log)
        log_input = tokenizer(raw['EventTemplate'], return_tensors="pt")
        log_input = {k:v[:, :512] for k, v in log_input.items()}  # max input length is 512
        log_output = bert_model(**log_input)
        log_vec = log_output.last_hidden_state.squeeze()[-1]  # used last state to instead of the text
        d[raw['EventTemplate']] = log_vec.detach().tolist()
    with open('./logs/Spark/event2vec.json', 'w') as f:
        json.dump(d, f)
        
@torch.no_grad()
def my_bert_embedding(logs_file: str):
    logs = load_logs(logs_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertSimilarity()
    model.load_state_dict(torch.load('./logs/Spark/bert.pth'))
    
    d = {}
    for index, raw in tqdm(logs.iterrows()):
        # print(log)
        log_input = tokenizer(raw['EventTemplate'], max_length=512, padding=True, truncation=True, return_tensors="pt")
       
        log_vec = model.forward_once(log_input)
        d[raw['EventTemplate']] = log_vec.squeeze().detach().tolist()
    with open('./logs/Spark/event2vec_mybert.json', 'w') as f:
        json.dump(d, f)

def load_json(fn:str) -> list:
    with open(fn, 'r') as f:
        lines = []
        for line in f.readlines():
            dic = json.loads(line)
            lines.append(dic)
        return lines

def save_data(data: list) -> None:
    pass

def split_data(data: list, train_val_test=(0.6, 0.1, 0.3)) -> dict:
    result = {}
    random.shuffle(data)  # shuffle
    total_count = len(data)
    train_ratio, val_ratio, test_ratio = train_val_test
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count
    result['train'] = copy.deepcopy(data[:train_count])
    result['val'] = copy.deepcopy(data[train_count: train_count + val_count])
    result['test'] = copy.deepcopy(data[train_count + val_count:])
    # save
    # print('save data...')
    # save_data(result['train'])
    # save_data(result['val'])
    # save_data(result['test'])

    print("total: {}, train/val/test: {}/{}/{}".format(total_count, train_count, val_count, test_count))

    return result

def load_logs(fn: str):
    df = pd.read_csv(fn)
    return df



def load_qa(qa_file:str) -> dict:
    qa_list = load_json(qa_file)
    datasets = split_data(qa_list)
    return datasets

if __name__ == '__main__':
    my_bert_embedding('./logs/Spark/spark_2k.log_templates.csv')