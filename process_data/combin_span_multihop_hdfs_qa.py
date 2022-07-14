import utils
import pandas as pd
import json
import sys


def read_json(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

span_path = '../logs/HDFS/hdfs_span_qa.json'
multihop_path = '../logs/HDFS/HDFS_multihop_qa.json'
structed_path = '../logs/HDFS/HDFS_2k.log_structured.csv'

structed_logs = pd.read_csv(structed_path)
span_data = utils.read_json(span_path)
multihop_data = list(utils.read_json(multihop_path))

span_transfer_data = []
for line in span_data:
    line['Logs'] = [line['RawLog']]
    line['Answer_type'] = 'Span'
    keywords = []
    q_token = line['Question'].lower().replace('?', '').replace('</s>', '').split()
    for token in q_token:
        if 'blk' in token:
            keywords.append(token)
        if 'verification' in token:
            keywords.append(token)
    line['keywords'] = keywords
    line['Events'] = [structed_logs[structed_logs['Content'] == line['RawLog']]['EventId'].values[0]]
    # print(line['Events'])
    line['Answer'] = line['Answer'].split(' ')[0]
    # 替换特殊字符
    context_token = line['RawLog'].lower().split()
    # print(context_token)
    for i, token in enumerate(context_token):
        if line['Answer'].lower() in token:
            line['answer_start'] = i
            break
    line['LogsCount'] = 1
    line.pop('RawLog')
    span_transfer_data.append(line)

# combine span and multihop to json
data = span_transfer_data + multihop_data

with open('../logs/HDFS/qa.json', 'w') as f:
    for line in data:
        print(line)
        f.write(json.dumps(line) + '\n')




