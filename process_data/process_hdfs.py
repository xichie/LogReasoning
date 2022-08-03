from utils import read_json
import json

qa_info = read_json('../logs/HDFS/qa.json')
with open('../logs/HDFS/qa_v2.json', 'w') as f:
    for line in qa_info:
        question = line['Question']
        answer = line['Answer']
        logs = line['Logs']
        anst = line['Answer_type']
        keywords = line['keywords']
        event = line['Events']
        answer_start = line['answer_start']
        count = line['LogsCount']

        new_line = {}
        new_line['Question'] = question.strip()
        if isinstance(answer, str):  # 如果答案是字符串， 只取:前面的部分
            new_line['Answer'] = answer.split(':')[0] # 只要IP, 不要端口
        else:
            new_line['Answer'] = answer
        
        new_line['Logs'] = logs
        new_line['Answer_type'] = anst
        new_line['keywords'] = keywords
        new_line['Events'] = event
        new_line['LogsCount'] = count
        log_tokens = logs[0].replace(':', ' ').split() # 日志种的冒号转为空格，分词
        for i, log_token in enumerate(log_tokens):
            if isinstance(new_line['Answer'], str):
                if new_line['Answer'] == log_token:
                    new_line['answer_start'] = i
                    break
            else:
                try:
                    log_token = float(log_token)
                    if log_token == new_line['Answer']:
                        new_line['answer_start'] = i
                        break
                except:
                    pass
        if 'answer_start' not in new_line:
            new_line['answer_start'] = -1

        f.write(json.dumps(new_line, ensure_ascii=False) + '\n')