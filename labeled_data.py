'''
    标记答案在日志模板中的位置
'''
from utils import read_json

def main():
    # 加载日志模板和qa数据
    templates_df = pd.read_csv('./logs/Spark/spark_2k.log_templates.csv')
    qa_data = read_json('./logs/Spark/spark_multihop_qa_v2.json')
    # 遍历qa数据, 标记答案在日志模板中的位置
    for i, qa_info in enumerate(qa_data):
        question = qa_info['Question']
        template = qa_info['Events'][0] # 因为答案只有一个事件, 所以取第一个事件即可
        eventTemplate = templates_df[templates_df['EventId'] == template]['EventTemplate'].values[0]
        