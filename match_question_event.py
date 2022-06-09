'''
    将问题匹配与日志模板匹配, 并计算准确率
'''
import torch
from model import BertSimilarity
from utils import read_json, get_similarity_logs
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def match_question_event(similarity_metric='Jaro'):
    log_events =  pd.read_csv('./logs/Spark/spark_2k.log_templates.csv')
    event2id = {row['EventTemplate']: row['EventId'] for index, row  in log_events.iterrows()}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if (similarity_metric in ['cosine', 'mybert'] )  else None
  
    if similarity_metric == 'cosine':
        bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif similarity_metric == 'mybert':
        bert_model = BertSimilarity()
        bert_model.load_state_dict(torch.load('./logs/Spark/bert.pth'))
    else:
        bert_model = None
        
    correct_count = 0
    total_count = 0
    for qa_info in  tqdm(read_json('./logs/Spark/spark_multihop_qa_v2.json')):
        most_similarity_events = get_similarity_logs(qa_info['Question'], log_events['EventTemplate'], similarity_metric, 'Spark', tokenizer, bert_model)
        most_similarity_eventIds = [event2id[event] for event in most_similarity_events]
        if most_similarity_eventIds[0] in qa_info['Events']:
            correct_count += 1
        total_count += 1

    return correct_count / total_count

if __name__ == '__main__':
    similarity_list = [
        # "random",
        # "Edit_Distance",
        # "jaccard",
        # "BM25",
        # "Jaro",
        # "jaro_winkler",
        # "cosine",
        'mybert',
    ]
    result = {'similarity_metric': [], 'accuracy': []}
    for similarity_metric in similarity_list:
        acc = match_question_event(similarity_metric)
        print('method: {}, accuarcy: {}'.format(similarity_metric ,acc))
        result['similarity_metric'].append(similarity_metric)
        result['accuracy'].append(acc)
        
    # pd.DataFrame(result).to_csv('./logs/Spark/spark_match_question_event_acc.csv')