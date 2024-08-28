import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# 定义推理函数
def predict_three_class(texts, model_path):
    """
    定义推理函数，用于对输入文本进行情感预测。

    Args:
        texts (list): 输入文本。
        model_path (str): 模型路径。

    Returns:
        DataFrame: 原始文本和预测结果，可以是"Positive"、"Negative"或"Neutral"。
    """
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 加载模型
    model = BertForSequenceClassification.from_pretrained(model_path)
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 将模型移动到相同的设备
    model.to(device)
    # 关闭模型的训练模式
    model.eval()
    # 预处理输入文本
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # 将输入数据移动到相同的设备
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # 关闭梯度计算以节省内存和计算力
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取预测结果
    predictions = torch.argmax(outputs.logits, dim=-1)
    # 初始化结果变量
    results = []
    for pred in predictions:
        if pred.item() == 0:
            results.append("Positive")
        elif pred.item() == 2:
            results.append("Negative")
        else:
            results.append("Neutral")
    # 创建一个DataFrame
    result_df = pd.DataFrame({
        'Texts': texts,
        'Sentiment_label': results
    })
    return result_df

model_path = './三分类model'
texts = ['这个电影很棒', '这个电影很差', '这个电影一般']
result_df = predict_three_class(texts, model_path)
print(result_df)