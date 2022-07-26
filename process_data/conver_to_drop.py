import json
import uuid

# 按行,读取json文件
def read_json(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

def convert_to_drop_format(qa_data, dataset, data_type):
    result = {}
    passages = [' '.join(qa_info['Logs']) for qa_info in qa_data]
    # passages = list(set(passages))
    print("passage number {}".format(len(passages)))
    for passage_idx, passage in enumerate(passages):
        result[passage_idx] = {}
        result[passage_idx]['passage'] = passage
        result[passage_idx]['qa_pairs'] = []
        # for qa_info in qa_data:
        qa_info = qa_data[passage_idx]
        rawlog = ' '.join(qa_info['Logs'])
        if rawlog == passage:
            qa_pair = {}
            qa_pair['question'] = qa_info['Question']
            qa_pair['answer'] = {'number': '', 
                                    'date': {
                                        'day': '', 
                                        'month': '', 
                                        'year': ''
                                        }, 
                                    'spans': []}
            if is_number(qa_info['Answer']):
                qa_pair['answer']['number'] = str(qa_info['Answer'])
            else:
                qa_pair['answer']['spans'].append(qa_info['Answer'])
            qa_pair['query_id']  = str(uuid.uuid1())
            result[passage_idx]['qa_pairs'].append(qa_pair)
 
    with open('../logs/{}/drop_{}.json'.format(dataset, data_type), 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    dataset = 'HDFS'
    convert_to_drop_format(read_json('../logs/{}/qa_train.json'.format(dataset)), dataset, 'train')
    convert_to_drop_format(read_json('../logs/{}/qa_test.json'.format(dataset)), dataset, 'test')
