from torch import nn
import torch
from dataloader import QADataset, MyDataLoader
from transformers import BertModel


'''
    基于bert finetune的问题匹配模型
'''

# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        euclidean_distance = torch.pairwise_distance(output1, output2)
        # cos_distance = torch.cosine_similarity(output1, output2) + 1.0
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class BertSimilarity(nn.Module):
    def __init__(self):
        super(BertSimilarity, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 768)
        self.clf = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 64),
        )
        self.sim_layer = nn.Linear(64*2, 1)
        self.softmax = nn.Softmax(dim=1)

        # 不冻结bert最后2层
        unfreeze_layers = ['layer.10', 'layer.11', 'pooler']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
                    # print(name) 
    
    def forward_once(self, input_token):
        embeds = self.bert(**input_token).last_hidden_state[:, -1] # (batch_size, 768)
        embeds = self.clf(embeds)  
        return embeds

    def forward(self, question_token, templates_token):
        q_embed = self.forward_once(question_token)
        t_embed = self.forward_once(templates_token)
    
        return q_embed, t_embed

# 训练模型
def train(model, dataloader, optimizer, criterion, device='cuda'):
    model.train()
    model = model.to(device)
    criterion = criterion.to(device)
    loss_total = 0
    for i, (questions, event, label) in enumerate(dataloader):
        for key in questions:
            questions[key] = questions[key].to(device)
        for key in event:
            event[key] = event[key].to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        q_embed, e_embed = model(questions, event)
        loss = criterion(q_embed, e_embed, label)
        loss.backward()
        optimizer.step()
        loss_total += loss.cpu().item()
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
    dataset = QADataset()
    dataloader = MyDataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    model = BertSimilarity()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = ContrastiveLoss()
    for epoch in range(50):
        loss = train(model, dataloader, optimizer, criterion)
        print('Epoch: %d, Loss: %.3f' % (epoch, loss))
        if (epoch+1) % 2 == 0:
            # 保存模型
            torch.save(model.state_dict(), './logs/Spark/bert.pth')

