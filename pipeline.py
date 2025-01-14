import time
import warnings
warnings.filterwarnings("ignore")

from Q2E import match_question_event
from QE2Log import rule_based_filter_spark, model_based_filter, rule_based_filter_hdfs, rule_based_filter_openssh, evaluate_match_qlogs_accuracy
from QEAnsPos import get_pos
from question_clf import evaluate
from utils import read_json, filter_digits
from tqdm import tqdm
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
    argparser.add_argument('--QE2Log', type=str, help='QE2Log to use, model or rule', default='Gold')
    arg = argparser.parse_args()
    dataset = arg.dataset
    QE2Log = arg.QE2Log

 
    q2e_acc, qe = match_question_event(dataset, 'mybert')  # (question, event)

    qe2log_start = time.time()
    if QE2Log == 'rule':
        if dataset == 'Spark':
            filter_logs =  rule_based_filter_spark(qe)  # (question, logs)
        elif dataset == 'HDFS':
            filter_logs =  rule_based_filter_hdfs(qe)  # (question, logs)
        elif dataset == 'OpenSSH':
            filter_logs =  rule_based_filter_openssh(qe)
        else:
            print('Dataset is invalid!!!')
            return
        evaluate_match_qlogs_accuracy(dataset, QE2Log)
    elif QE2Log == 'model':
        filter_logs = model_based_filter(dataset, qe)
        evaluate_match_qlogs_accuracy(dataset, QE2Log)
    else:   # Gold retrival
        filter_logs = {}   
        for qa_info in tqdm(read_json('./logs/{}/qa_test.json'.format(dataset))):
            filter_logs[qa_info['Question']] = qa_info['Logs']
    qe2log_end = time.time()
    start = time.time()
    position_dict = get_pos(qe, dataset) # (question, pos)

    ############################Log Reasoner#####################################
    '''
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
            if dataset == 'Spark':
                log_token = log.replace(',', '').replace(')', '').split()
            elif dataset == 'HDFS':
                log_token = log.replace(':', ' ').split()
            elif dataset == 'OpenSSH':
                log_token = log.replace(':', ' ').replace('(', ' ').replace(')', ' ').replace('=', ' ').split()
            else:
                print('Dataset was not found!!!')
            # print(log_token)
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
            pred_ans = float(pred_ans)
            ans = float(qa_info['Answer'])
            if pred_ans == ans:
                correct += 1
            else:
                print(i, 'False1', ans, '---', pred_ans)
        except:
            ans = str(qa_info['Answer'])
            pred_ans = str(pred_ans)
            if ans in pred_ans:
                correct += 1
            else:
                print(i, 'False2', ans, '---', pred_ans)
        total += 1
        answer_list.append((qa_info['Question'], pred_ans))

    print("EM:{}".format(correct/total))
    end = time.time()
    print("Log reasoning time:", (end-start))
    
    print("Log qe2log time:", (qe2log_end-qe2log_start))
    with open('./results/{}/result.csv'.format(dataset), 'w') as f:
        for q, a in answer_list:
            f.write(q + ', ' + str(a) + '\n')

    '''
if __name__ == '__main__':
    main()
    # q2e_acc, qe = match_question_event('mybert')  # (question, event)
    # filter_logs =  rule_based_filter_hdfs(qe)
    # # filter_logs =  rule_based_filter(qe)  # (question, logs)
    # filter_logs =  model_based_filter(qe)  # (question, logs)