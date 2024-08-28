import os
import torch
import jieba
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

# 合并指定目录中所有Excel文件的数据，并返回一个DataFrame
def merge_excel_to_dataframe(directory_path):
    """
    合并指定目录中所有Excel文件的数据。
    
    参数:
    directory_path: 字符串, 表示包含Excel文件的目录路径。
    
    返回:
    pandas.DataFrame, 包含合并后所有Excel文件的数据。
    """
    # 初始化一个空的DataFrame，用于存储合并后的数据
    merged_df = pd.DataFrame()

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件扩展名是否为xlsx或xls
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            # 构建完整的文件路径
            file_path = os.path.join(directory_path, filename)
            # 读取Excel文件，假设所有文件的字段都相同且在第一行
            df = pd.read_excel(file_path)
            # 将读取的数据追加到merged_df中
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df

# 合并指定目录中所有CSV文件的数据，并返回一个DataFrame
def merge_csv_to_dataframe(directory_path):
    """
    合并指定目录中所有CSV文件的数据。
    
    参数:
    directory_path: 字符串, 表示包含CSV文件的目录路径。
    
    返回:
    pandas.DataFrame, 包含合并后所有CSV文件的数据。
    """
    # 初始化一个空的DataFrame，用于存储合并后的数据
    merged_df = pd.DataFrame()

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            # 使用chunksize分批读取文件
            chunk_size = 10000  # 根据内存大小调整
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # 将读取的块追加到merged_df中
                merged_df = pd.concat([merged_df, chunk], ignore_index=True)

    return merged_df

# DataFrame数据处理
def dataframe_process(df, row_name, max_length:int):
    """
    DataFrame数据预处理
    """
    # 去除重复和空值
    df = df.drop_duplicates(subset=[row_name])
    df = df.dropna(subset=[row_name])
    
    # 截断文本长度
    def truncate_text(text):
        if len(text) > max_length:
            return text[:max_length]
        return text
    df[row_name] = df[row_name].apply(truncate_text)
    
    # df = df[df["评论类型"]=='评论主体']
    df = df[df["用户等级"]>=5]
    df = df[df['评论内容'].str.contains(r'[\u4e00-\u9fa5]{5,}')]    # 删除评论内容少于5个中文字符的评论
    df['评论内容'] = df['评论内容'].str.replace(r'\[\s*.*?\s*\]', '', regex=True)   # 删除评论内容中的表情
    
    return df

# 获得分词文件：对DataFrame中的文本列进行分词和过滤，并保存结果至文本文件中。
def dataframe_text_cut(df, row_name, userdict_path:str, stopword_path:str, save_path:str):
    """
    对DataFrame中的文本列进行分词和过滤，并保存结果至文本文件中（去除停用词、数字和长度小于2的词）。

    参数:
    df: 包含文本数据的DataFrame
    row_name: 存储文本数据的列名
    userdict_path: 自定义词典的路径，用于增加分词的准确性
    stopword_path: 停用词列表的路径，用于过滤常见无意义词
    save_path: 分词后结果的保存路径
    
    返回:
    None
    """
    # 将文本列转换为列表形式
    text_list = df[row_name].tolist()
    # 加载自定义词典，提高分词准确性
    jieba.load_userdict(userdict_path)
    # 读取停用词列表，用于过滤常见无意义词
    stopwords = [line.strip() for line in open(stopword_path, encoding='UTF-8').readlines()]
    
    # 存储分词并过滤后的文本
    cutted_text = []
    for i, line_text in enumerate(text_list):
        # 对文本进行分词
        seg_list = jieba.cut(line_text)
        # 过滤停用词、数字和长度小于2的词
        filtered_words = []
        for word in seg_list:
            # 判断是否为停用词、数字和长度小于2的词
            if (word not in stopwords and (len(word)) >= 2 and not word.isdigit()):
                filtered_words.append(word)
        # 将过滤后的词组重新组合成字符串
        cutted_text.append(' '.join(filtered_words))
        
    # 将分词并过滤后的文本结果保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cutted_text))

# 获得向量文件：使用SentenceTransformer模型对文本数据集中的文本进行嵌入处理
def embeddingByST(df, row_name, model_name, output_file):
    """
    使用SentenceTransformer模型对文本数据集中的文本进行嵌入处理，并保存嵌入向量。

    参数:
    df: pandas.DataFrame
        包含文本数据的DataFrame，其中一列名为row_name包含待处理的文本。
    row_name: str
        DataFrame中包含文本的列的名称。
    model_name: str
        SentenceTransformer模型的名称，用于加载预训练模型。
    output_file: str
        保存嵌入向量的文件路径，嵌入向量将以numpy数组的形式保存。

    返回:
    成功提示
    """
    # 将文本列转换为列表形式
    text_list = df[row_name].tolist()
    # 加载指定的SentenceTransformer模型
    embedding_model = SentenceTransformer(model_name)
    # 对输入的文本列表进行嵌入编码，显示进度条
    embeddings = embedding_model.encode(text_list, show_progress_bar=True)
    # 打印嵌入向量的类型和形状信息
    print(type(embeddings), embeddings.shape)
    # 将嵌入向量保存到指定的输出文件
    np.save(output_file, embeddings)
    
    print("完成embedding")

# 获得向量文件：使用BERT模型对文本数据集中的文本进行嵌入处理
def embeddingByBERT(df, row_name, model_name, batch_size:int, output_file):
    """
    获得向量文件：使用BERT模型对文本数据集中的文本进行嵌入处理
    
    """
    text = df[row_name].tolist()
    
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    #模型使用GPU加速
    #torch.cuda.is_available()  检测是否可用GPU加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #将模型设置为评估模式
    model.eval()

    #切分数据
    data_loader = DataLoader(text, batch_size=batch_size)

    # ---- 文本转向量 ----
    # 生成的向量存放在这里
    cls_embeddings = []

    # 使用tqdm显示处理进度
    # tqdm b站教程：https://www.bilibili.com/video/BV1ZG411M7Ge/?spm_id_from=333.337.search-card.all.click&vd_source=eace37b0970f8d3d597d32f39dec89d8
    for batch_sentences in tqdm(data_loader):
        
        # tokenizer官方文档：https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
        # truncation=True，对输入句子进行截断，这里确保最大长度不超过512个字
        # max_length：不设置的话，默认会截断到该模型可接受的最大长度
        # padding=True 或 padding='longest': 将所有句子填充到批次中最长句子的长度
        # padding="max_length": 将所有句子填充到由 max_length 参数指定的长度
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
        # print(123, inputs.input_ids[0], tokenizer.decode(inputs.input_ids[0]))
        
        # 把编码好的数据，也放在device上，It is necessary to have both the model, and the data on the same device, either CPU or GPU
        # https://huggingface.co/docs/transformers/v4.39.2/en/main_classes/tokenizer#transformers.BatchEncoding.to
        # https://stackoverflow.com/questions/63061779/pytorch-when-do-i-need-to-use-todevice-on-a-model-or-tensor
        inputs.to(device)

        # 设置不要计算梯度
        # 一般来说，如果我们只是用模型进行“预测”，而不涉及对模型进行更新时，就不需要计算梯度，以此来节约内存，增加运算效率
        # with上下文中，对model的调用将遵循torch.no_grad()，即不会计算梯度
        with torch.no_grad():
            outputs = model(**inputs)

        # 把这一批词向量存入cls_embeddings容器中
        # tensor.cpu() 将张量移动到 CPU
        # tensor.numpy() 将 CPU 上的张量转换为 NumPy 数组
        cls_embeddings.append(outputs.last_hidden_state[:, 0].cpu().numpy()) # 只取CLS对应的向量

        # print('pt格式', type(outputs.last_hidden_state[:, 0].shape), outputs.last_hidden_state[:, 0].shape)
        # print('numpy格式', type(outputs.last_hidden_state[:, 0].cpu().numpy()), outputs.last_hidden_state[:, 0].cpu().numpy().shape)

    # 合并句子向量
    print(f'batch个数（批数）：{len(cls_embeddings)}，每批的向量维度：{cls_embeddings[0].shape}，最后一批的向量维度：{cls_embeddings[-1].shape}，每批向量的类型：{type(cls_embeddings[0])}')
    cls_embeddings_np = np.vstack(cls_embeddings)
    print('最终生成的词向量', type(cls_embeddings_np), cls_embeddings_np.shape)

    # ---- 保存词嵌入向量 ----
    # 保存句子向量到npy文件
    # 官方文档：https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(output_file, cls_embeddings_np)
    print("词向量存储于: ", output_file)

