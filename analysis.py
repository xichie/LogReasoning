'''
数据分析
'''

from calendar import c
from cmath import log
import utils

# 分析训练集测试集事件出现的次数
def analysis_event_count(dataset):
    qa_train = utils.read_json('logs/{}/qa_train.json'.format(dataset))
    qa_test = utils.read_json('logs/{}/qa_test.json'.format(dataset))

    train_counter = {}
    test_counter = {}
    for qa in qa_train:
        event = qa['Events'][0]
        if event in train_counter.keys():
            train_counter[event] += 1
        else:
            train_counter[event] = 1
    for qa in qa_test:
        event = qa['Events'][0]
        if event in test_counter.keys():
            test_counter[event] += 1
        else:
            test_counter[event] = 1
    print('训练集事件出现的次数：')
    print(train_counter)
    print('测试集事件出现的次数：')
    print(test_counter)
    print('测试集中的事件在训练集中没有出现：')
    for event in test_counter.keys():
        if event not in train_counter.keys():
            print(event)

# 问题类型的分析
def count_word4question(dataset):
    qa_data = utils.read_json('logs/{}/qa.json'.format(dataset))

    counter = {}
    total = 0
    for qa in qa_data:
        question = qa['Question']
        q_token = question.replace('?', '').split()[0]
   
        if q_token not in counter.keys():
            counter[q_token] = 1
        else:
            counter[q_token] += 1
        total += 1
    print(counter)
    for k, v in counter.items():
        print(k, v/total * 100)


# 分析回答问题所需日志的数量
def analysis_log_count(dataset):
    qa_data = utils.read_json('logs/{}/qa.json'.format(dataset))

    total = 0
    log_count = 0
    for qa in qa_data:
        ans_type = qa['Answer_type']
        if not ans_type == 'Span':
            log_count += len(qa['Logs'])
            total += 1
    print('{}回答问题(不包含span类型)平均所需日志的数量：'.format(dataset))
    print(log_count/total)
    

if __name__ == "__main__":
    dataset = 'OpenSSH'
    # analysis_event_count(dataset)
    # count_word4question(dataset)
    analysis_log_count(dataset)