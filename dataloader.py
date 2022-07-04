from nis import cat
import torch
from torch.utils.data import Dataset, DataLoader
from utils import read_json, read_csv
import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np

'''
    输出应该是一个三元组
    (question_token, templates_token, label)
    question_token: 问题bert模型的输入
    templates_token: 一个随机的日志事件
    label: 0代表正例(两者应该相似), 1代表负例(两者不相似)
'''


# 定义一个类，继承Dataset
class QADataset(Dataset):
    def __init__(self):
        # 加载qa数据
        qa_data = read_json('./logs/HDFS/hdfs_multihop_qa_train.json')
        # 加载所有的日志事件
        log_templates = pd.read_csv('logs/HDFS/HDFS_2k.log_templates.csv')
        self.templates = list(log_templates['EventTemplate'])
        self.events_count = len(log_templates)
        # Bert tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.templates_token = tokenizer(self.templates, max_length=512, padding=True, truncation=True, return_tensors='pt')
        # 日志事件 -> index
        self.eventId2index = {row['EventId'] : index for index, row in log_templates.iterrows()}
        
        questions = []
        events = []
        for qa in qa_data: 
            questions.append(qa['Question'])
            events.append(self.eventId2index[qa['Events'][0]])

        self.q_token = tokenizer(questions, max_length=512, padding=True, truncation=True, return_tensors='pt')
        self.len = len(questions)
        self.events = events
        
    def __getitem__(self, index):
        batch_size = index.stop - index.start
        
        pos_num = int(batch_size * 0.5) # 正类样本数量
        neg_num = batch_size - pos_num # 负类样本数量
        
        q_token_ = {}
        for key in self.q_token.keys():
            q_token_[key] = self.q_token[key][index]

        # 随机挑选neg_num个日志事件, 作为负类样本
        neg_index = np.random.randint(0, self.events_count, neg_num)
        pos_index = self.events[index][neg_num:]
        selected_index = np.concatenate((neg_index, pos_index))
        
        t_token = {}
        for key in self.templates_token.keys():
            t_token[key] = self.templates_token[key][selected_index]
        # print(np.array(self.events[index]))
        # print(np.array((selected_index)))
        # print(np.array(self.events[index]) != np.array((selected_index)))
        label = torch.LongTensor(np.array(self.events[index]) != np.array((selected_index)))
      
        return q_token_, t_token,  label
    
    def __len__(self):
        return self.len

# 定义一个类，继承DataLoader
class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        super(MyDataLoader, self).__init__(dataset, batch_size, shuffle, num_workers=num_workers, drop_last=True)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i * self.batch_size: (i + 1) * self.batch_size]


if __name__ == '__main__':
    
    dataset = QADataset()
    dataloader = MyDataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    for i, (question, event, label) in enumerate(dataloader):
        print(question['input_ids'].size())
        print(event['input_ids'].size())
        print(label.size())
        