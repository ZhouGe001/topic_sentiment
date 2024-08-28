from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def word_frequency(filename):
    """
    统计给定文件中每个单词出现的频率。
    
    参数:
    filename (str): 需要进行单词频率统计的文件的路径。
    
    返回:
    dict: 包含每个单词及其出现次数的字典。
    """
    # 打开文件并读取内容
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    # 使用空格分割字符串，得到单词列表
    words = text.split()
    # 创建一个空字典来存储词频
    word_counts_dict = {}
    # 遍历单词列表，统计每个词出现的次数
    for word in words:
        # 如果单词已存在于字典中，其计数增加1；否则，将其添加到字典并设置计数为1
        if word in word_counts_dict:
            word_counts_dict[word] += 1
        else:
            word_counts_dict[word] = 1
    return word_counts_dict

def generate_wordcloud(word_counts_dict):
    """
    生成词云图
    
    Args:
        word_counts_dict (dict): 词频字典，键为词，值为词频
        
    Returns:
        None
    """
    # 计算词频
    word_counts_dict = Counter(word_counts_dict)
    # 创建词云对象
    wordcloud = WordCloud(
        font_path=r'C:/Windows/Fonts/simhei.ttf',  # 指定中文字体路径
        width=800, 
        height=400, 
        background_color='white'
    ).generate_from_frequencies(word_counts_dict)

    # 显示词云图
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 不显示坐标轴
    plt.show()