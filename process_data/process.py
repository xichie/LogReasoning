import json
from tqdm import tqdm
import pandas as pd
from utils import read_json, generate_uuid

'''
    日志multihop qa数据v4 (最终版) 
'''
def save_multihop_qa():
    qa_info = read_json('logs/Spark/spark_multihop_questions.json')
    qa_info_v3 = read_json('logs/Spark/spark_multihop_qa_v3.json')
    f = open('logs/Spark/spark_multihop_qa_v4.json', 'a+')
    for qa1, qa2 in tqdm(zip(qa_info_v3, qa_info)):
        qa1['Answer_type'] = qa2['Answer_type']
        qa1['keywords'] = qa2['keywords']
        f.write(json.dumps(qa1, ensure_ascii=False) + '\n')
        

'''
    划分训练测试集
'''
def split_train_test(dataset):
    questions = []
    with open('../logs/{}/qa.json'.format(dataset)) as f:
        for line in f.readlines():
            questions.append(json.loads(line))
            
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(questions, test_size=0.3, random_state=1)
    print('train:', len(train))
    print('test:', len(test))
    with open('../logs/{}/qa_train.json'.format(dataset), 'w') as f:
        for line in train:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    with open('../logs/{}/qa_test.json'.format(dataset), 'w') as f:
        for line in test:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

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
    提取问题, 做为Numerical Reasoning的输入
'''
def save_question(multihop_qa_data, dataset, data_type):
    # multihop_qa_data = read_json('./logs/Spark/spark_multihop_qa_v3.json')
    with open('../logs/{}/questions_{}.json'.format(dataset, data_type), 'w') as f:
        for qa_info in multihop_qa_data:
            line = {}
            q_token = qa_info['Question'].replace('?', '').replace(':', ' ').replace('/', ' ').split()
            
            line['Question'] = q_token
            line['keywords'] = qa_info['keywords']
            line['Answer_type'] = qa_info['Answer_type']
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
def transfer2SquAD(qa_data, dataset, data_type='train'):
    templates_df = pd.read_csv('../logs/{}/{}_2k.log_templates.csv'.format(dataset, dataset))
    # 转化为字典, key为事件id, value为事件模板
    templates_dict = {}
    for index, row in templates_df.iterrows():
        templates_dict[row['EventId']] = row['EventTemplate']
    squad_data = []
    for idx, line in enumerate(qa_data):
        
        question = line['Question']
        answer_idx = line['answer_start']
        eventID = line['Events'][0]
        template = templates_dict[eventID]
        
        template_token = template.replace('/', ' ').replace(':', ' ').split(' ')
        answer_start = 0
        
        if answer_idx == -1:  # 答案是计数类型
            answer_text = ''
        else:    
            for idx, token in enumerate(template_token):
                if idx == answer_idx:
                    break
                answer_start += len(token) + 1
            print(question)
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
    with open('../logs/{}/squad_{}.json'.format(dataset, data_type), 'w') as f:
        f.write(json.dumps(squad_data, ensure_ascii=False) + '\n')
        

'''
    转为SQuAD for other model, 只考虑Span类型
''' 
def transfer2SquAD_v2(qa_data, dataset, data_type='train'):
    squad_data = []
    for idx, line in enumerate(qa_data):
        
        question = line['Question']
        answer_idx = line['answer_start']
        answer_type = line['Answer_type']
        answer_start = 0
        
        if answer_type == 'Span':  # 答案是Span类型
            answer_text = line['Answer']
            log = line['Logs'][0]
            log_token = log.split()
            for idx, token in enumerate(log_token):
                if idx == answer_idx:
                    break
                answer_start += len(token) + 1
            squad_data.append({
                'title': '',
                'paragraphs': [
                    {
                        'context': log,
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
    with open('../logs/{}/squad_{}_v2.json'.format(dataset, data_type), 'w') as f:
        f.write(json.dumps(squad_data, ensure_ascii=False) + '\n')
        

if __name__ == '__main__':

    dataset = "Spark"

    # transfer_rawlog_to_logs()
    # labeled_question_position()
    # split_train_test(dataset)
    # transfer2SquAD(read_json('../logs/{}/qa_{}.json'.format(dataset, 'train')), dataset, 'train')
    # transfer2SquAD(read_json('../logs/{}/qa_{}.json'.format(dataset, 'test')), dataset, 'test')
    transfer2SquAD_v2(read_json('../logs/{}/qa_{}.json'.format(dataset, 'train')), dataset, 'train')
    transfer2SquAD_v2(read_json('../logs/{}/qa_{}.json'.format(dataset, 'test')), dataset, 'test')
    # save_question(read_json('../logs/{}/qa_{}.json'.format(dataset, 'train')), dataset, 'train')
    # save_question(read_json('../logs/{}/qa_{}.json'.format(dataset, 'test')), dataset, 'test')
    # labeled_question_keyword()
    # save_multihop_qa()