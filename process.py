import json
import pandas as pd
from utils import read_json
from collections import Counter
import uuid


'''
    将qa中的rawlogs转换为logs,并添加对应的EventID 后续应该用不到, 应该直接使用structured_logs的结果去创建qa data.
'''
def transfer_rawlog_to_logs():
    log_structured = pd.read_csv('./logs/Spark/spark_2k.log_structured.csv')
    # 读取json文件，保存到新文件
    with open('./logs/Spark/spark_multihop_qa_v2.json', 'w') as f:
        for qa_info in read_json('./logs/Spark/spark_multihop_qa.json'):
            qa_info['Logs'] = [] 
            qa_info['Events'] = []
            for raw_log in qa_info['RawLog']:
                log = raw_log.split(':')[-1].strip()
                qa_info['Logs'].append(log)
                for index, row in log_structured.iterrows():
                    if log in row['Content']:
                        qa_info['Events'].append(row['EventId'])
                        break
            qa_info['LogsCount'] = len(qa_info['Logs'])
            del qa_info['RawLog']
            del qa_info['RawLog_count']
            
            f.write(json.dumps(qa_info, ensure_ascii=False) + '\n')

'''
    标记答案在日志模板中的位置
'''
def labeled_question_position():
    # 加载日志模板和qa数据
    templates_df = pd.read_csv('./logs/Spark/spark_2k.log_templates.csv')
    qa_data = read_json('./logs/Spark/spark_multihop_qa_v2.json')
    # 遍历qa数据, 标记答案在日志模板中的位置
    for i, qa_info in enumerate(qa_data):
        if i >= 31:
            question = qa_info['Question']
            template = qa_info['Events'][0] # 因为答案只有一个事件, 所以取第一个事件即可
            eventTemplate = templates_df[templates_df['EventId'] == template]['EventTemplate'].values[0]
            print(eventTemplate)
            # 分词, 得到每个token的位置信息
            token_pos = [(idx, token) for idx, token in enumerate(eventTemplate.split(' '))]    
        
            print('Question:', question)
            print('EventTemplate', eventTemplate)
            print(token_pos)
            answer_start = int(input('请输入答案起始位置:\n'))
            qa_info['answer_start'] = answer_start
            
            # 保存qa数据
            with open('./logs/Spark/spark_multihop_qa_v3.json', 'a') as f:
                f.write(json.dumps(qa_info, ensure_ascii=False) + '\n')
            print('保存了{}条数据'.format(i+1))


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans
 
'''
    转为SQuAD格式
''' 
def transfer2SquAD():
    templates_df = pd.read_csv('./logs/Spark/spark_2k.log_templates.csv')
    # 转化为字典, key为事件id, value为事件模板
    templates_dict = {}
    for index, row in templates_df.iterrows():
        templates_dict[row['EventId']] = row['EventTemplate']
    
    squad_data = []
    raw_data = read_json('./logs/Spark/spark_multihop_qa_v3.json')
    for idx, line in enumerate(raw_data):
        
        question = line['Question']
        answer_idx = line['answer_start']
        eventID = line['Events'][0]
        template = templates_dict[eventID]
        
        template_token = template.split(' ')
        answer_start = 0
        
        if answer_idx == -1:  # 答案是计数类型
            answer_text = ''
        else:    
            for idx, token in enumerate(template_token):
                if idx == answer_idx:
                    break
                answer_start += len(token) + 1
            answer_text = template_token[answer_idx]
            

        squad_data.append({
            'title': '',
            'paragraphs': [
                {
                    'context': template,
                    'qas': [
                        {
                            'answers': [
                                {
                                    'answer_start': answer_start,
                                    'text': answer_text,
                                },
                            ],
                            'question': question,
                            'id': generate_uuid('')    
                        }
                        
                    ]
                }
            ]
            
        })
    squad_data = {'data': squad_data}
    with open('./logs/Spark/spark_multihop_qa_squad.json', 'w') as f:
        f.write(json.dumps(squad_data, ensure_ascii=False) + '\n')
        
# 生成UUID
def generate_uuid(prefix):
    uuid_ = prefix + '-' + str(uuid.uuid4())
    return uuid_       

def split_train_test():
    with open('./logs/Spark/spark_multihop_qa_squad.json') as f:
        raw_data = json.load(f)
    
    data = raw_data['data']
    
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    print('train:', len(train))
    print('test:', len(test))
    with open('./logs/Spark/spark_multihop_squad_train.json', 'w') as f:
        train_data = {'data': train}
        f.write(json.dumps(train_data, ensure_ascii=False) + '\n')
    with open('./logs/Spark/spark_multihop_squad_test.json', 'w') as f:
        test_data = {'data': test}
        f.write(json.dumps(test_data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # transfer_rawlog_to_logs()
    # labeled_question_position()
    # transfer2SquAD()
    split_train_test()