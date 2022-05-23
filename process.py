import json
import pandas as pd
from utils import read_json

'''
    将qa中的rawlogs转换为logs,并添加对应的EventID 后续应该用不到, 应该直接使用structured_logs的结果去创建qa data.
'''
def transfer_rawlog_to_logs():
    log_structured = pd.read_csv('./logs/Spark/spark_2k.log_structured.csv')
    # 读取json文件，保存到新文件
    with open('./logs/Spark/spark_multihop_qa_v2.json', 'w') as f:
        for qa_info in read_json('./logs/Spark/spark_multihop_qa.json'):
            qa_info['Logs'] = [] 
            qa_info['Events'] = []
            for raw_log in qa_info['RawLog']:
                log = raw_log.split(':')[-1].strip()
                qa_info['Logs'].append(log)
                for index, row in log_structured.iterrows():
                    if log in row['Content']:
                        qa_info['Events'].append(row['EventId'])
                        break
            qa_info['LogsCount'] = len(qa_info['Logs'])
            del qa_info['RawLog']
            del qa_info['RawLog_count']
            
            f.write(json.dumps(qa_info, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    transfer_rawlog_to_logs()
