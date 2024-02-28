
# LogReasoning
## Pipeline
Script Execution: run.sh

1. Train the Question-Event Matching Model: `python q2e_model.py`

![Accuracy Plot](results/Spark/spark_match_question_event_acc.png)

2. Save embeddings corresponding to each event: `python bert_embedding.py`
3. Evaluate the results of the Question-Event Matching Model: `python Q2E.py`
4. Filter original logs based on events matched with each question and save filtered QA results: `python QE2Log_model.py` (model-based)

5. Train QANet to extract answer positions from events: `cd QANet-pytorch-/`, then `python main.py --mode data` and `python main.py --mode train`
5. Evaluate QANet results and save positions of answers for each question in log events: `cd QANet-pytorch-/`, then `python main.py --mode test`
6. Extract answers to questions based on results from steps 4 and 5
7. Determine computations to be performed based on questions

# LogReasoning
## Pipline
运行脚本: run.sh

1. 训练问题-事件匹配模型: python q2e_model.py 

<img src="results/Spark/spark_match_question_event_acc.png" width=300> \

2. 保存每个事件对应的embedding: python bert_embedding.py
3. 评估问题-事件匹配模型的的结果: python Q2E.py
4. 根据每个问题匹配的事件, 过滤原始logs, 保存过滤后的QA结果  python QE2Log_model.py(model-based)

5. 训练QANet提取事件中答案的位置: cd QANet-pytorch-/ python main.py --mode data   python main.py --mode train
5. 评估QANet的结果，并保存每个问题对应答案在日志事件中的位置: cd QANet-pytorch-/ python main.py --mode test
6. 根据4、5的结果提取问题的答案
7. 根据问题判断应该进行的计算
