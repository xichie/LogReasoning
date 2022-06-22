'''
    问题分类
'''
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from utils import read_json
import json


class QuestionDataset(Dataset):
    def __init__(self, type='train'):
        self.type = type
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        qa_data = read_json('./logs/Spark/spark_multihop_questions_{}.json'.format(type))
        
        answer_type_mapping = {
            'addition': 0,
            'count': 1,
            'maximum': 2,
            'minimum': 3
        }
        examples = {
            'tokens': [],
            'ner_tags': [],
            'logs': [],
            'answer_type': [],
        }
        for qa_info in qa_data:
            question = qa_info['Question']
            keywords = qa_info['keywords']
            logs = qa_info['Logs']
            answer_type = qa_info['Answer_type']
            label = [0] * len(question)
            for kw in keywords:
                idx = question.index(kw)
                label[idx] = 1
            
            examples['tokens'].append(question)
            examples['ner_tags'].append(label)
            examples['logs'].append(logs)
            examples['answer_type'].append(answer_type_mapping[answer_type])
            
        self.examples = examples
        if type == 'train':
            self.tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)
            self.tokenized_inputs['answer_type'] = self.examples['answer_type']
            for k, v in self.tokenized_inputs.items():
                self.tokenized_inputs[k] = torch.LongTensor(v)  
        else:
            self.tokenized_inputs = self.examples
    def __getitem__(self, index):
        if self.type == 'train':
            return {k: v[index] for k, v in self.tokenized_inputs.items()}
        return {k: v[index] for k, v in self.examples.items()}
    def __len__(self):
        return len(self.examples['tokens'])

class QuestionDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i * self.batch_size: (i + 1) * self.batch_size]

class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
        # 只训练分类器
        unfreeze_layers = ['layer.5', 'classifier']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
                    # print(name) 
    def forward(self, inputs):
        output = self.bert(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['answer_type'])
        return output

# 训练模型
def train(model, dataloader, optimizer, device='cuda'):
    model.train()
    model = model.to(device)
    loss_total = 0
    for i, examples in enumerate(dataloader):
        examples_ = {}
        for key in examples:
            examples_[key] = examples[key].to(device)
        optimizer.zero_grad()
        output = model(examples_)
        # print(output)
        loss = output.loss
        loss.backward()
        optimizer.step()
        loss_total += loss.cpu().item()
    return loss_total / len(dataloader)

@torch.no_grad()
def evaluate(model=None):
    print('Evaluating...')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if model == None:
        model = QModel()
        model.load_state_dict(torch.load('./logs/Spark/BertForSeqClf.pth'))
        model = model.cuda()
        
    model.eval()
    dataloader = QuestionDataLoader(QuestionDataset('test'), batch_size=32)
    
    total, correct, loss = 0, 0, 0
    pred_label_list = []
    for i, examples in enumerate(dataloader):
        inputs = tokenizer(examples['tokens'], truncation=True, padding=True, is_split_into_words=True)
        inputs['answer_type'] = examples['answer_type']
        for k, v in inputs.items():
            inputs[k] = torch.LongTensor(v).cuda()
        output = model(inputs)
        loss += output.loss.cpu().item()
        logits = output.logits   # (bs, 4)
        # print(logits.size())
        pred_label = torch.argmax(logits, dim=1).cpu().numpy()
        pred_label_list.extend(list(pred_label.squeeze()))
        for i, label in enumerate(examples['answer_type']):
            if label == pred_label[i]:
                correct += 1
            total += 1
    print('Correct/Total:{}/{}, Accuracy:{}'.format(correct, total, correct/total))
    return loss, pred_label_list

if __name__ == '__main__':
    train_loader = QuestionDataLoader(QuestionDataset(type='train'), batch_size=16)
    model = QModel()
    train_loader = QuestionDataLoader(QuestionDataset(type='train'), batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(100):
        model.train()
        train_loss = train(model, train_loader, optimizer)
        test_loss = 0
        if (epoch+1) % 5 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
            # 保存模型
            torch.save(model.state_dict(), './logs/Spark/BertForSeqClf.pth')
            # 评估模型
            test_loss, pred_label = evaluate(model)
            print('Epoch:{}, Train Loss:{}, Test Loss:{}'.format(epoch, train_loss, test_loss))
        print('Epoch: %d, Train Loss: %.3f' % (epoch, train_loss))
    