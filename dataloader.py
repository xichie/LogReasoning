import re
import torch
from torch.utils.data import Dataset, DataLoader
from utils import read_json, read_csv
import pandas as pd
# 读取qa数据
def load_data(file_path):
    qa_data = read_json(file_path)
    return qa_data




# 定义一个类，继承Dataset
class QADataset(Dataset):
    def __init__(self, qa_data):
        log_templates = pd.read_csv('logs/Spark/spark_2k.log_templates.csv')
        eventId2index = {row['EventId'] : index for index, row in log_templates.iterrows()}
        self.question = []
        self.event = []
        for qa in qa_data: 
            self.question.append(qa['Question'])
            self.event.append(eventId2index[qa['Events'][0]])
        self.len = len(self.question)
        # self.question = torch.Tensor(self.question)
        self.event = torch.LongTensor(self.event)
    def __getitem__(self, index):
        return self.question[index], self.event[index]
    def __len__(self):
        return self.len

# 定义一个类，继承DataLoader
class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        super(MyDataLoader, self).__init__(dataset, batch_size, shuffle, num_workers=num_workers)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i * self.batch_size: (i + 1) * self.batch_size]


if __name__ == '__main__':
    qa_data = load_data('./logs/Spark/spark_multihop_qa_v2.json')
    dataset = QADataset(qa_data)
    dataloader = MyDataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    for i, (question, event) in enumerate(dataloader):
        print(question)
        print(event)
        break