import warnings
warnings.filterwarnings("ignore")

from Q2E import match_question_event
from QE2Log import rule_based_filter_spark, model_based_filter, rule_based_filter_hdfs
from QEAnsPos import get_pos
from question_clf import evaluate
from utils import read_json, filter_digits
import argparse

'''
answer_type_mapping = {
            'addition': 0,
            'count': 1,
            'maximum': 2,
            'minimum': 3,
            'span': 4,
        }
'''
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use')
    arg = argparser.parse_args()
    dataset = arg.dataset

    q2e_acc, qe = match_question_event(dataset, 'mybert')  # (question, event)
    
    filter_logs =  rule_based_filter_hdfs(qe)  # (question, logs)
    # filter_logs = model_based_filter(dataset, qe)
    position_dict = get_pos(qe) # (question, pos)

    # AnsPos2Num
    i = 1
    pred_qa = []
    for question, logs in filter_logs.items():
        ans = []
        if len(logs) == 0:
            i += 1
            pred_qa.append((question, ans))
            continue
        for log in logs:
            log_token = log.replace(',', '').replace(')', '').split()
            idx = position_dict[str(i)][0]
            ans.append(log_token[idx])
        pred_qa.append((question, ans))
        i += 1
    
    
    # 判断答案类型
    _, answer_type = evaluate(dataset)
    total, correct = 0, 0
    answer_list = []
    for i, qa_info in enumerate(read_json('./logs/{}/qa_test.json'.format(dataset))):
        pred_ans = '---'
        ans_list = pred_qa[i][1]
        anst = answer_type[i]
   
        if anst == 0: # add
            ans_list = filter_digits(ans_list)
            if len(ans_list) > 0:
                ans_list = list(map(float, ans_list))
                pred_ans = sum(ans_list)
        elif anst == 1: # count
            pred_ans = len(ans_list)
        elif anst == 2: # maximum
            ans_list = filter_digits(ans_list)
            if len(ans_list) > 0:
                ans_list = list(map(float, ans_list))
                pred_ans = max(ans_list)
        elif anst == 3: # min
            ans_list = filter_digits(ans_list)
            if len(ans_list) > 0:  
                ans_list = list(map(float, ans_list))
                pred_ans = min(ans_list)
        elif anst == 4: # Span
            if len(ans_list) > 0:
                pred_ans = ans_list[0]
        else:
            pred_ans = '---'
        try:
            if len(filter_logs[i]) == len(qa_info['Logs']) or pred_ans == float(qa_info['Answer']):
                correct += 1
                print(i, 'correct')
        except:
            if str(pred_ans) == str(qa_info['Answer']):
                correct += 1
                print(i, 'correct')
        total += 1
        answer_list.append((qa_info['Question'], pred_ans))

    print("EM:{}".format(correct/total))

    with open('./results/{}/result.csv'.format(dataset), 'w') as f:
        for q, a in answer_list:
            f.write(q + ', ' + str(a) + '\n')

    
if __name__ == '__main__':
    main()
    # q2e_acc, qe = match_question_event('mybert')  # (question, event)
    # filter_logs =  rule_based_filter_hdfs(qe)
    # # filter_logs =  rule_based_filter(qe)  # (question, logs)
    # filter_logs =  model_based_filter(qe)  # (question, logs)