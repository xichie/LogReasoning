'''
    将问题匹配与日志模板匹配, 并计算准确率
'''
import torch
from model import BertSimilarity
from utils import read_json, get_similarity_logs
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from collections import Counter
import string
import json

# 根据问题匹配的事件，过滤日志 (rule-based)
def filter_logs_by_event(similarity_metric='mybert'):
    acc, qe = match_question_event(similarity_metric)
    df = pd.read_csv('./logs/Spark/spark_2k.log_structured.csv')
    exclude = [')', '(', ',', ':', '', '?']  # 将文本中的特殊符号替换
    
    filter_logs = {}
    for q, e in qe.items():
        filter_logs[q] = []
        logs = df[df['EventId'] == e]['Content'].to_list()
        # 计算问题中每个单词在logs中出现的次数
        q_ = ''.join([c for c in q if c not in exclude])
        q_ = q_.replace('_', ' ')
        q_token = q_.split(' ')
        counter = {token: 0 for token in q_token}
        for log in logs:
            log = ''.join(ch for ch in log if ch not in exclude)
            log = log.replace('_', ' ')
            log_token = log.split()
            for token in q_token:
                if token in log_token:
                    counter[token] += 1
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
         
        q_token = []
        for token, count in counter:
            if count == 0:
                continue
            q_token.append(token)
        if len(q_token) == 0:
            continue
        for token in q_token[:-1]:
            for log in logs:
                log_ = ''.join(ch for ch in log if ch not in exclude)
                log_ = log_.replace('_', ' ')
                substr = token + ' ' + q_token[-1]
                log_token = log_.split()
                for i in range(len(log_token) - 1):
                    if log_token[i] == token and log_token[i + 1] == q_token[-1] and substr in q:
                        filter_logs[q].append(log)
    
    # 保存结果               
    f = open('logs/Spark/spark_question_logs_filter.json', 'w')
    for q, logs in filter_logs.items():
        q_logs = json.dumps({'Question': q, 'Logs': logs})
        f.write(q_logs + '\n')
    f.close()

# 评估问题匹配日志的准确率
def evaluate_match_qlogs_accuracy():
    results = []
    with open('./logs/Spark/spark_question_logs_filter.json', 'r') as f:
        for line in f:
            q_logs = json.loads(line)
            results.append(q_logs)
    # print(results[31])
    with open('./logs/Spark/spark_multihop_qa_v3.json', 'r') as f:
        idx = 0
        err_count = 0
        for line in f:
            qa_info = json.loads(line)
            if len(results[idx]['Logs']) == 0:
                err_count += 1
                print(idx)
            else:    
                for log in results[idx]['Logs']:
                    if log not in qa_info['Logs']:
                        err_count += 1
                        # print(log)
                        # print(qa_info['Logs'])
                        print(idx)
                        break
            idx += 1
        print('accuracy:', 1 - err_count / len(results))
        
def match_question_event(similarity_metric='Jaro'):
    log_events =  pd.read_csv('./logs/Spark/spark_2k.log_templates.csv')
    event2id = {row['EventTemplate']: row['EventId'] for index, row  in log_events.iterrows()}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if (similarity_metric in ['cosine', 'mybert'] )  else None
  
    if similarity_metric == 'cosine':
        bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif similarity_metric == 'mybert':
        bert_model = BertSimilarity()
        bert_model.load_state_dict(torch.load('./logs/Spark/bert.pth'))
    else:
        bert_model = None
        
    correct_count = 0
    total_count = 0
    qe = {} # question: event
    for qa_info in  tqdm(read_json('./logs/Spark/spark_multihop_qa_v3.json')):
        if qa_info['Question'] in qe.keys():
                print(qa_info['Question'])
        most_similarity_events = get_similarity_logs(qa_info['Question'], log_events['EventTemplate'], similarity_metric, 'Spark', tokenizer, bert_model)
        most_similarity_eventIds = [event2id[event] for event in most_similarity_events]
        if most_similarity_eventIds[0] in qa_info['Events']:  # 如果问题匹配到了正确事件，则添加到qe中
            correct_count += 1
            qe[qa_info['Question']] = most_similarity_eventIds[0]
        else:                                                # 否则，问题对应的事件为空
            qe[qa_info['Question']] = ''                
        total_count += 1
    acc = correct_count / total_count
    return acc, qe 

if __name__ == '__main__':
    similarity_list = [
        # "random",
        # "Edit_Distance",
        # "jaccard",
        # "BM25",
        # "Jaro",
        # "jaro_winkler",
        # "cosine",
        'mybert',
    ]
    # result = {'similarity_metric': [], 'accuracy': []}
    # for similarity_metric in similarity_list:
    #     acc, qe = match_question_event(similarity_metric)
    #     print('method: {}, accuarcy: {}'.format(similarity_metric ,acc))
    #     result['similarity_metric'].append(similarity_metric)
    #     result['accuracy'].append(acc)
        
    # pd.DataFrame(result).to_csv('./logs/Spark/spark_match_question_event_acc.csv')
    
    # filter_logs_by_event()
    evaluate_match_qlogs_accuracy()