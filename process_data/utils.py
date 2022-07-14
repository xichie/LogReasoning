import json
import random
import Levenshtein as le
# from gensim.summarization import bm25
import numpy as np
from collections import Counter
from transformers import BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity
import torch
import uuid


def isNum(e):
    try:
        e = float(e)
        return True
    except:
        return False

# 过滤list中数字
def filter_digits(a:list) -> list:
    b = []
    for e in a:
        try:
            e = float(e)
            b.append(e)
        except:
            pass
    return b
    


# 生成UUID
def generate_uuid(prefix):
    uuid_ = prefix + '-' + str(uuid.uuid4())
    return uuid_   

# 按行,读取json文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

# 读取csv文件
def read_csv(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip().split(',')



def cosine_similarity(x,y):
    x = np.array(x)
    y = np.array(y)
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

@torch.no_grad()
def bert_method(question: str, logs: list, dataset: str, tokenizer, bert_model) -> list:
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # bert_model = BertModel.from_pretrained("bert-base-uncased")

    with open('./logs/{}/event2vec.json'.format(dataset),'r')as f:
        log2vec = json.load(f)

    question_input = tokenizer(question, return_tensors="pt")
    question_output = bert_model(**question_input)
    question_vec = question_output.last_hidden_state.squeeze()[-1].detach().tolist()
    logs_vec = []
    for log in logs:
        logs_vec.append(log2vec[log])
    similarity_list = []
    for i, log_vec in enumerate(logs_vec):
        score = cosine_similarity(question_vec, log_vec)
        similarity_list.append((score, logs[i]))
    return similarity_list

class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.75):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()
 
    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))
 
    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))
 
        return score
 
    def get_documents_score(self, query, reverse=True):
        score_list = []
        for i in range(self.documents_number):
            doc = self.documents_list[i]
            socre = self.get_score(i, query)
            score_list.append((socre, ' '.join(doc)))
        score_list = sorted(score_list, key=lambda x: x[0], reverse=reverse)
        
        return score_list

def jaccard_similarity(a, b):
    # convert to set
    a = set(a)
    b = set(b)
    # calucate jaccard similarity
    j = float(len(a.intersection(b))) / len(a.union(b))
    return j

def get_topk_similarity_logs(question: str, logs: list, top_k: int, similarity:str) -> list:
    metrics = {
        "Edit Distance": {"func": le.distance,"reverse": False}, # lower is best
        "jaccard": {"func": jaccard_similarity, "reverse": True},  # higher is best
        "BM25": {"func": None, "reverse": True},  # higher is best
        "Jaro": {"func": le.jaro, "reverse": True},  # higher is best
        "jaro_winkler": {"func": le.jaro_winkler, "reverse": True}, # higher is best
        "consine": {"func": None, "reverse": True}, # higher is best
    }

    if similarity == 'random':
        random.shuffle(logs)
        return logs[:top_k]

    if similarity == 'BM25':
        docs = []
        for log in logs:
            docs.append(log.split())
        # model = BM25_Model(docs)
        # topk_similarity_pairs = model.get_documents_score(question, metrics[similarity]['reverse'])[:top_k]
        # topk_similarity_logs = [pair[1] for pair in topk_similarity_pairs]
        model = bm25.BM25(docs)
        score_list = []
        for i, log in enumerate(logs):
            score = model.get_score(question, i)
            score_list.append((score, log))
        score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
        topk_similarity_logs = [pair[1] for pair in score_list[:top_k]]
        return topk_similarity_logs

    if similarity == 'cosine':
        score_list = bert_method(question, logs)
        score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
        topk_similarity_logs = [pair[1] for pair in score_list[:top_k]]

        return topk_similarity_logs

    # other similarity metric     
    similarity_list = []
    sim = metrics[similarity]['func']
    for log in logs:
        similarity_score = sim(question, log)
        similarity_list.append((similarity_score, log))
    topk_similarity_pairs = sorted(similarity_list, key=lambda x: x[0], reverse=metrics[similarity]['reverse'])[:top_k]
    topk_similarity_logs = [pair[1] for pair in topk_similarity_pairs]
    return topk_similarity_logs

def get_similarity_logs(question: str, logs: list, similarity:str, dataset: str='', tokenizer=None, bert_model=None) -> list:
    metrics = {
        "Edit_Distance": {"func": le.distance,"reverse": False}, # lower is best
        "jaccard": {"func": jaccard_similarity, "reverse": True},  # higher is best
        "BM25": {"func": None, "reverse": True},  # higher is best
        "Jaro": {"func": le.jaro, "reverse": True},  # higher is best
        "jaro_winkler": {"func": le.jaro_winkler, "reverse": True}, # higher is best
        "consine": {"func": None, "reverse": True}, # higher is best
        "mybert": {"func": None, "reverse": True}, # higher is best
    }

    if similarity == 'random':
        random.shuffle(logs)
        return logs

    if similarity == 'BM25':
        docs = []
        for log in logs:
            docs.append(log.split())
        model = BM25_Model(docs)
        similarity_pairs = model.get_documents_score(question, metrics[similarity]['reverse'])
        similarity_logs = [pair[1] for pair in similarity_pairs]
        
        # model = bm25.BM25(docs)
        # score_list = []
        # for i, log in enumerate(logs):
        #     score = model.get_score(question, i)
        #     score_list.append((score, log))
        # score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
        # similarity_logs = [pair[1] for pair in score_list]
        return similarity_logs

    if similarity == 'cosine':
        score_list = bert_method(question, logs, dataset, tokenizer, bert_model)
        score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
        similarity_logs = [pair[1] for pair in score_list]

        return similarity_logs
    
    if similarity == 'mybert':
        score_list = my_bert(question, logs, dataset, tokenizer, bert_model)
        score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
        similarity_logs = [pair[1] for pair in score_list]

        return similarity_logs

    # other similarity metric     
    similarity_list = []
    sim = metrics[similarity]['func']
    for log in logs:
        similarity_score = sim(question, log)
        similarity_list.append((similarity_score, log))
    topk_similarity_pairs = sorted(similarity_list, key=lambda x: x[0], reverse=metrics[similarity]['reverse'])
    similarity_logs = [pair[1] for pair in topk_similarity_pairs]
    return similarity_logs

@torch.no_grad()
def my_bert(question: str, logs: list, dataset: str='', tokenizer=None, bert_model=None) -> list:
    question_input = tokenizer(question, max_length=512, padding=True, truncation=True, return_tensors="pt")
    question_vec = bert_model.forward_once(question_input)
    question_vec = question_vec.detach().numpy()
    
    logs_vec = []
    if not dataset == '': 
        with open('./logs/{}/event2vec_mybert.json'.format(dataset),'r')as f:
            log2vec = json.load(f)    
        for log in logs:
            logs_vec.append(log2vec[log])
    else:
        for log in logs:
            log_input = tokenizer(log, max_length=512, padding=True, truncation=True, return_tensors="pt")
            log_vec = bert_model.forward_once(log_input)
            log_vec = log_vec.detach().numpy()
            logs_vec.append(log_vec)
        
    similarity_list = []
    for i, log_vec in enumerate(logs_vec):
        score = cosine_similarity(question_vec, log_vec)
        similarity_list.append((score, logs[i]))
        
    return similarity_list


if __name__ == '__main__':
    # b = filter_digits(['RE', '1', '2.0', 32])
    # print(b)
    print(isNum('127.0'))
