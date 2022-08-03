# Logreasoning runing script
dataset=Spark
echo "========================Training Q2E model on $dataset======================================"
python q2e_model.py --dataset $dataset
python bert_embedding.py --dataset $dataset
echo "========================Training QE2Log model on $dataset======================================"
python QE2Log_model.py --dataset $dataset
echo "========================Training NumReasoning model on $dataset======================================"
python question_clf.py --dataset $dataset
echo "========================Training QEAnsPos model on $dataset======================================"
cd QANet-pytorch-
python main.py --mode data --dataset $dataset
python main.py --mode train --dataset $dataset
cd ..
cp QANet-pytorch-/log/*.json logs/$dataset/
echo "========================Evaluate on $dataset======================================"
python pipeline.py --dataset $dataset 