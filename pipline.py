from q2e import match_question_event, filter_logs_by_event
from QEAnsPos import get_pos
from question_clf import evaluate
from utils import read_json, filter_digits

'''
answer_type_mapping = {
            'addition': 0,
            'count': 1,
            'maximum': 2,
            'minimum': 3
        }
'''
def main():
    q2e_acc, qe = match_question_event('mybert')  # (question, event)
    filter_logs =  filter_logs_by_event(qe)  # (question, logs)
    position_dict = get_pos(qe) # (question, pos)
    # AnsPos2Num
    i = 1
    pred_qa = []
    for question, logs in filter_logs.items():
        ans = []
        if len(logs) == 0:
            i += 1
            print(i, '---------', ans)
            pred_qa.append((question, ans))
            continue
        for log in logs:
            log_token = log.replace(',', '').replace(')', '').split()
            idx = position_dict[str(i)][0]
            ans.append(log_token[idx])
        pred_qa.append((question, ans))
        print(i, '---------', ans)
        i += 1
    # 判断答案类型
    _, answer_type = evaluate()
    total, correct = 0, 0
    for i, qa_info in enumerate(read_json('./logs/Spark/spark_multihop_qa_test.json')):
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
        else:
            pred_ans = ''
        print(pred_ans)
        try:
            if pred_ans == float(qa_info['Answer']):
                correct += 1
        except:
            if str(pred_ans) == str(qa_info['Answer']):
                correct += 1
        total += 1
        
    print("EM:{}".format(correct/total))
    
if __name__ == '__main__':
    main()