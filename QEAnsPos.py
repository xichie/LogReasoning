import json

def get_pos(qe:dict):
    with open('QANet-pytorch-/log/answers_idx.json') as f:
        ans_position = json.load(f)
    return ans_position
        
        
    
    