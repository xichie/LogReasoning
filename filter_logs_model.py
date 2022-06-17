import torch
from torch import nn
from utils import read_json
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import json

def split_train_test():
    questions = []
    with open('./logs/Spark/spark_multihop_questions.json') as f:
        for line in f.readlines():
            questions.append(json.loads(line))

    
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(questions, test_size=0.1, random_state=1)
    print('train:', len(train))
    print('test:', len(test))
    with open('./logs/Spark/spark_multihop_questions_train.json', 'w') as f:
        for line in train:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    with open('./logs/Spark/spark_multihop_questions_test.json', 'w') as f:
        for line in test:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

'''
    标记日志中的关键词(基于模型的预测)
'''
class DistilbertForTokenClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        
        # 只训练分类器
        unfreeze_layers = ['layer.5', 'classifier']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
                    # print(name) 
            
                    
    def forward(self, inputs):
        output = self.model(**inputs)
        return output
        
class QuestionDataset(Dataset):
    def __init__(self, type='train'):
        self.type = type
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        qa_data = read_json('./logs/Spark/spark_multihop_questions_{}.json'.format(type))
        examples = {
            'tokens': [],
            'ner_tags': []
        }
        for qa_info in qa_data:
            question = qa_info['Question']
            keywords = qa_info['keywords']
            label = [0] * len(question)
            for kw in keywords:
                idx = question.index(kw)
                label[idx] = 1
            
            examples['tokens'].append(question)
            examples['ner_tags'].append(label)
            
        self.examples = examples
        if type == 'train':
            self.tokenized_inputs = self.tokenize_and_align_labels(examples, self.tokenizer)
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
    
    @staticmethod  
    def tokenize_and_align_labels(examples, tokenizer):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)
        
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        for k, v in tokenized_inputs.items():
            tokenized_inputs[k] = torch.LongTensor(v)
        return tokenized_inputs

class QuestionDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i * self.batch_size: (i + 1) * self.batch_size]


# 训练模型
def train(model, dataloader, optimizer, device='cuda'):
    model.train()
    model = model.to(device)
    loss_total = 0
    for i, examples in enumerate(dataloader):
        for key in examples:
            examples[key] = examples[key].to(device)
            
        optimizer.zero_grad()
        output = model(examples)
        loss = output.loss
        loss.backward()
        optimizer.step()
        loss_total += loss.cpu().item()
    return loss_total / len(dataloader)

@torch.no_grad()
def evaluate(model):
    print('Evaluating...')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if model == None:
        model = DistilbertForTokenClassification()
        model.load_state_dict(torch.load('./logs/Spark/distilBert.pth'))
        model = model.cuda()
        
    model.eval()
    dataloader = QuestionDataLoader(QuestionDataset('test'), batch_size=32)
    
    total, correct, loss = 0, 0, 0
    for i, examples in tqdm(enumerate(dataloader)):
        inputs = QuestionDataset.tokenize_and_align_labels(examples, tokenizer)
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        output = model(inputs)
        loss += output.loss.cpu().item()
        logits = output.logits
        pred_label = torch.argmax(logits, dim=2).cpu().numpy()
    
        for i, label in enumerate(examples['ner_tags']):
            length = len(label)
            label = np.array(label)
            
            if (pred_label[i][1:length+1] == label).all():
                correct += 1
            else:
                # print(label)
                # print(pred_label[i][1:length+1])
                # print(' '.join(examples['tokens'][i]))
                pass
            total += 1
    
    print('Correct/Total:{}/{}, Accuracy:{}'.format(correct, total, correct/total))
    return loss
    
if __name__ == '__main__':
    # split_train_test()
    model = DistilbertForTokenClassification()
    train_loader = QuestionDataLoader(QuestionDataset(type='train'), batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(100):
        model.train()
        train_loss = train(model, train_loader, optimizer)
        test_loss = 0
        if (epoch+1) % 20 == 0 and epoch > 50 :
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
            # 保存模型
            torch.save(model.state_dict(), './logs/Spark/distilBert.pth')
            # 评估模型
            test_loss = evaluate(model)
            print('Epoch:{}, Train Loss:{}, Test Loss:{}'.format(epoch, train_loss, test_loss))
        print('Epoch: %d, Train Loss: %.3f' % (epoch, train_loss))
    # evaluate()
    