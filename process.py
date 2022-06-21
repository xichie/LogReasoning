import json
from lib2to3.pgen2 import token
import pandas as pd
from utils import read_json, generate_uuid


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


'''
    提取问题
'''
def save_question():
    qa_data = read_json('./logs/Spark/spark_multihop_qa_v3.json')
    with open('./logs/Spark/spark_multihop_questions.json', 'a') as f:
        for qa_info in qa_data:
            line = {}
            q_token = qa_info['Question'].replace('?', '').split()
            line['Question'] = q_token
            jieci = ['of', 'by', 'in']
            line['keywords'] = []
            for jc in jieci:
                if jc in q_token:
                    idx = q_token.index(jc)
                    line['keywords'] = q_token[idx+1: ]
                    break
            line['Answer_type'] = ''
            line['Logs'] = qa_info['Logs']
            
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

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
        
    

# 划分训练集和测试集
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
    # split_train_test()
    # labeled_question_keyword()
    save_question()