from utils import read_json, get_similarity_logs
from filter_logs_model import DistilbertForTokenClassification, evaluate
import torch
import pandas as pd
import json

# model-based
def model_based_filter(qe):
    filter_logs = {}
    loss, key_words_all = evaluate()
    print(key_words_all)
    df = pd.read_csv('./logs/HDFS/HDFS_2k.log_structured.csv')

    for i, (q, e) in enumerate(qe.items()):
        filter_logs[q] = []
        logs = df[df['EventId'] == e]['Content'].to_list() # 对应事件的日志
        for log in logs:
            log_lower = log.lower() # log转为小写
            kw_include = []
            for kw in key_words_all[i]:
                if kw in log_lower:
                    kw_include.append(kw)
            if len(kw_include) == 0:
                continue
            # if len(kw_include) == len(key_words_all[i]):  # 所有关键字都在log中，则判断该子字符串是否包含(HDFS)
            #     if ' '.join(kw_include) in log:
            #         filter_logs[q].append(log)
            if len(kw_include) > 2 or len(kw_include) == len(key_words_all[i]):
                filter_logs[q].append(log)
                 
                
  
    f = open('logs/HDFS/hdfs_question_logs_filter_model_based.json', 'w')
    for q, logs in filter_logs.items():
        q_logs = json.dumps({'Question': q, 'Logs': logs})
        f.write(q_logs + '\n')
    f.close()
    return filter_logs

# (HDFS)根据问题匹配的事件，过滤日志 (rule-based)
def rule_based_filter(qe) -> dict:
    df = pd.read_csv('./logs/Spark/spark_2k.log_structured.csv')
    exclude = [')', '(', ',', ':', '', '?']  # 将文本中的特殊符号替换
    
    filter_logs = {}
    for i, (q, e) in enumerate(qe.items()):
        filter_logs[q] = []
        logs = df[df['EventId'] == e]['Content'].to_list() # 对应事件的日志
        # 计算问题中每个单词在logs中出现的次数
        q_ = ''.join([c for c in q if c not in exclude])
        # q_ = q_.replace('_', ' ')
        q_token = q_.split()
        counter = {token: 0 for token in q_token}
        for log in logs:
            for token in q_token:
                if token in log:
                    counter[token] += 1
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True) # 排序
        
        # 将问题中出现次数不为0的单词作为过滤条件
        q_token = []
        for token, count in counter:
            if count == 0:
                continue
            q_token.append(token)
        if len(q_token) == 0:  # 由于没有匹配到正确的事件，所以直接返回空列表
            pass
            # print(e)
            # print(q)
        elif 'rdd' in q_token[-1]:  # 包含rdd的关键字的日志
            for log in logs:
                if q_token[-1] in log:
                    filter_logs[q].append(log)
        else:   # 次数不为0的关键字两两组合     
            for token in q_token[:-1]:
                for log in logs:
                    log_ = ''.join(ch for ch in log if ch not in exclude)
                    log_ = log_.replace('_', ' ')
                    substr = token + ' ' + q_token[-1]
                    log_token = log_.split()
                    for i in range(len(log_token) - 1):
                        if log_token[i] == token and log_token[i + 1] == q_token[-1] and substr in q:
                            filter_logs[q].append(log)
        if len(filter_logs[q]) == 0:  # 没有通过关键字过滤，则不过滤
            filter_logs[q] = logs
    
    # 保存结果               
    f = open('logs/Spark/spark_question_logs_filter_rule_based.json', 'w')
    for q, logs in filter_logs.items():
        q_logs = json.dumps({'Question': q, 'Logs': logs})
        f.write(q_logs + '\n')
    f.close()
    return filter_logs

def rule_based_filter_hdfs(qe) -> dict:
    df = pd.read_csv('./logs/HDFS/HDFS_2k.log_structured.csv')
    filter_logs = {}
    for i, (q, e) in enumerate(qe.items()):
        filter_logs[q] = []
        logs = df[df['EventId'] == e]['Content'].to_list() # 对应事件的日志
        # 计算问题中每个单词在logs中出现的次数
        q_ = q.replace(':', ' ').replace('/', ' ').replace('?', '').lower()
        q_token = q_.strip().split()
        counter = {token: 0 for token in q_token}
        for log in logs:
            for token in q_token:
                if token in log.lower():
                    counter[token] += 1
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True) # 排序
        print(counter)
        # 将问题在log中出现次数不为0的单词作为过滤条件
        q_token = []
        for token, count in counter:
            if count == 0:
                continue
            q_token.append(token)
        if len(q_token) == 0:  # 由于没有匹配到正确的事件，所以直接返回空列表
            pass
        else:
            for log in logs:
                flag = True
                for token in q_token:
                    if token not in log.lower() and token not in ['is', 'of', 'in', 'by', 'the', 'a', 'an', 'can', 'be', 'on', 'from', 'to']:
                        flag = False
                if flag:
                    filter_logs[q].append(log) 
        if len(filter_logs[q]) == 0:  # 没有通过关键字过滤，则不过滤
            filter_logs[q] = logs
        
    # 保存结果               
    f = open('logs/HDFS/hdfs_question_logs_filter_rule_based.json', 'w')
    for q, logs in filter_logs.items():
        q_logs = json.dumps({'Question': q, 'Logs': logs})
        f.write(q_logs + '\n')
    f.close()
    return filter_logs   
            
# 评估问题匹配日志的准确率
def evaluate_match_qlogs_accuracy():
    results = []
    with open('./logs/HDFS/hdfs_question_logs_filter_model_based.json', 'r') as f:
        for line in f.readlines():
            q_logs = json.loads(line)
            results.append(q_logs)
    print(results[1])
    with open('./logs/HDFS/hdfs_multihop_qa_test.json', 'r') as f:
        idx = 0
        err_count = 0
        for line in f.readlines():
            qa_info = json.loads(line)
            if len(results[idx]['Logs']) == 0:
                err_count += 1
                print(idx, '--')
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

if __name__ == '__main__':
    # model_based_filter(None)
    evaluate_match_qlogs_accuracy()