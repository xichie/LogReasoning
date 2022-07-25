from QE2Log_model import evaluate
import pandas as pd
import json
import argparse

# model-based
def model_based_filter(dataset, qe):
    stop_words = ['contain', 'free', 'is', 'of', 'in', 'by', 'the', 'a', 'an', 'can', 'be', 'on', 'from', 'to', 'take', 'up', 'free', 'that', 'for']
    filter_logs = {}
    loss, key_words_all = evaluate(dataset)
    for i in range(len(key_words_all)):
        key_words_all[i] = [word for word in key_words_all[i] if word not in stop_words]
    print(key_words_all)
    df = pd.read_csv('./logs/{}/{}_2k.log_structured.csv'.format(dataset, dataset))

    for i, (q, e) in enumerate(qe.items()):
        filter_logs[q] = []
        logs = df[df['EventId'] == e]['Content'].to_list() # 对应事件的日志
        cur_kw_list = [] 
        for kw in key_words_all[i]:
            if 'rdd' in kw:
              cur_kw_list.append(kw)
              break
        if len(cur_kw_list) == 0:
            cur_kw_list = key_words_all[i]
        for log in logs:
            log_lower = log.lower() # log转为小写
            if len(cur_kw_list) == 0:
                continue
            if dataset == 'Spark':
                if ' '.join(cur_kw_list) in log_lower:  
                    filter_logs[q].append(log)
            elif dataset == 'HDFS':
                log_token = log_lower.split()
                log_flag = True  # log是否包含所有关键字
                # 判断所有kw都在log中
                for kw in cur_kw_list:
                    kw_flag = False
                    for token in log_token:
                        if kw in token:
                            kw_flag = True
                            break
                    if not kw_flag:
                        log_flag = False
                        break   
                if log_flag:
                    filter_logs[q].append(log)

            
    f = open('logs/{}/filter_model_based.json'.format(dataset), 'w')
    for q, logs in filter_logs.items():
        q_logs = json.dumps({'Question': q, 'Logs': logs})
        f.write(q_logs + '\n')
    f.close()
    return filter_logs

# 根据问题匹配的事件，过滤日志 (rule-based)
def rule_based_filter_spark(qe) -> dict:
    df = pd.read_csv('./logs/Spark/Spark_2k.log_structured.csv')
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
        elif 'broadcast' in  q_token[-1]:  # 包含broadcast的关键字的日志
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
    f = open('logs/Spark/filter_rule_based.json', 'w')
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
    f = open('logs/HDFS/filter_rule_based.json', 'w')
    for q, logs in filter_logs.items():
        q_logs = json.dumps({'Question': q, 'Logs': logs})
        f.write(q_logs + '\n')
    f.close()
    return filter_logs   
            
# 评估问题匹配日志的准确率
def evaluate_match_qlogs_accuracy(dataset, QE2Log):
    results = []
    if QE2Log == 'model':
        with open('./logs/{}/filter_model_based.json'.format(dataset), 'r') as f:
            for line in f.readlines():
                q_logs = json.loads(line)
                results.append(q_logs)
    else:
        with open('./logs/{}/filter_rule_based.json'.format(dataset), 'r') as f:
            for line in f.readlines():
                q_logs = json.loads(line)
                results.append(q_logs)
    # print(results[1])
    with open('./logs/{}/qa_test.json'.format(dataset), 'r') as f:
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
        print('QE2Log accuracy ({}): {}'.format(QE2Log, 1 - err_count / len(results)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use')
    argparser.add_argument('--QE2Log', type=str, help='QE2Log to use, model or rule')
    arg = argparser.parse_args()
    dataset = arg.dataset
    QE2Log = arg.QE2Log

    # q2e_acc, qe = match_question_event(dataset, 'mybert')  # (question, event)
    # model_based_filter(dataset, qe)
    evaluate_match_qlogs_accuracy(dataset)