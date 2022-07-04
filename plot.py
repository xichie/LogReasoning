from turtle import color
import matplotlib.pyplot as plt
import pandas as pd

'''
可视化结果
'''

def plot_match_question_event_acc():
    df = pd.read_csv('./results/HDFS/hdfs_match_question_event_acc.csv')
    plt.bar(df['similarity_metric'], df['accuracy'])
    
    # 显示数字
    for a, b in zip(df['similarity_metric'], df['accuracy']):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=6)
    
    plt.title('HDFS Match Question Event Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('./results/HDFS/hdfs_match_question_event_acc.png')

if __name__ == '__main__':
    plot_match_question_event_acc()