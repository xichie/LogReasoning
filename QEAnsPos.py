import json

def get_pos(qe:dict, dataset):
    with open('./logs/{}/answers_idx.json'.format(dataset)) as f:
        ans_position = json.load(f)
    return ans_position
        
        
    
    