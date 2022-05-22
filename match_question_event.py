'''
    将问题匹配与日志模板匹配
'''
from utils import read_json, get_similarity_logs
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def match_question_event(similarity_metric='Jaro'):
    log_events =  pd.read_csv('./logs/Spark/spark_2k.log_templates.csv')
    event2id = {row['EventTemplate']: row['EventId'] for index, row  in log_events.iterrows()}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if similarity_metric == 'cosine' else None
    bert_model = BertModel.from_pretrained("bert-base-uncased") if similarity_metric == 'cosine' else None

    correct_count = 0
    for qa_info in  tqdm(read_json('./logs/Spark/spark_multihop_qa_v2.json')):
        most_similarity_events = get_similarity_logs(qa_info['Question'], log_events['EventTemplate'], similarity_metric, 'Spark', tokenizer, bert_model)
        most_similarity_eventIds = [event2id[event] for event in most_similarity_events]
        if most_similarity_eventIds[0] in qa_info['Events']:
            correct_count += 1
    
    print('correct_count: ', correct_count)

if __name__ == '__main__':
    similarity_list = [
        # "random",
        # "Edit_Distance",
        # "jaccard",
        # "BM25",
        # "Jaro",
        # "jaro_winkler",
        "cosine",
    ]
    for similarity_metric in similarity_list:
        match_question_event(similarity_metric)
