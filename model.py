from statistics import mode
from torch import nn
import torch
import pandas as pd
from dataloader import QADataset, MyDataLoader, load_data
from transformers import BertTokenizer, BertModel



class BertSimilarity(nn.Module):
    def __init__(self):
        super(BertSimilarity, self).__init__()
        self.templates = list(pd.read_csv('logs/Spark/spark_2k.log_templates.csv')['EventTemplate'])
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 768)
        self.softmax = nn.Softmax(dim=1)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x1):
        batch_size = len(x1)
        x1_token = self.tokenizer(x1, max_length=512, padding=True, truncation=True, return_tensors='pt')
        x1_embeds = self.bert(**x1_token).last_hidden_state[:, -1] # (batch_size, 768)
        x1_embeds = self.dropout(x1_embeds)
        x1_embeds = self.linear(x1_embeds)

        x2_token = self.tokenizer(self.templates, max_length=512, padding=True, truncation=True, return_tensors='pt')
        x2_embeds = self.bert(**x2_token).last_hidden_state[:, -1] # (batch_size, 768)
        x2_embeds = self.dropout(x2_embeds)
        x2_embeds = self.linear(x2_embeds)

        similarity = torch.zeros(batch_size, len(self.templates)) # (batch_size, num_templates)
        for i in range(batch_size):
            similarity[i] = torch.cosine_similarity(x1_embeds[i].unsqueeze(0), x2_embeds) # 35
        similarity_prob = self.softmax(similarity)
        return similarity_prob

# 训练模型
def train(model, dataloader, optimizer, criterion, device='cuda'):
    model.train()
    model = model.to(device)
    loss_total = 0
    for i, (question, event) in enumerate(dataloader):
        event = event.cuda()
        optimizer.zero_grad()
        similarity_prob = model(question)
        loss = criterion(similarity_prob, event)
        loss.backward()
        optimizer.step()
        loss_total += loss.cpu.item()
    return loss_total / len(dataloader)

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    model = model.to(device)
    loss_total = 0
    with torch.no_grad():
        for i, (question, event) in enumerate(dataloader):
            similarity_prob = model(question)
            similarity_prob.cpu().argmax(axis=1)  
    return loss_total / len(dataloader)

if __name__ == '__main__':
    qa_data = load_data('./logs/Spark/spark_multihop_qa_v2.json')
    dataset = QADataset(qa_data)
    dataloader = MyDataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    model = BertSimilarity()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        loss = train(model, dataloader, optimizer, criterion)
        print('Epoch: %d, Loss: %.3f' % (epoch, loss))
        if (epoch+1) % 2 == 0:
            # 保存模型
            torch.save(model.state_dict(), './logs/Spark/bert.pth')

