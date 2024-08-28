from datasets import load_dataset
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn

# 定义一个自定义的BERT分类模型
class BertForSequenceClassificationCustom(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BertForSequenceClassificationCustom, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        return (loss, logits) if loss is not None else logits

# 加载tokenizer和数据集
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = load_dataset("tyqiangz/multilingual-sentiments", 'chinese')

# 定义tokenizer的处理函数
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

# 定义一个计算模型性能指标的函数
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='macro')
    acc = accuracy_score(p.label_ids, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results-三分类',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
)

# 创建一个带有填充功能的数据合并器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 实例化自定义模型
model = BertForSequenceClassificationCustom('bert-base-chinese', num_labels=3)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 评估模型
trainer.evaluate()

# 保存模型
model.save_pretrained('./三分类model')
tokenizer.save_pretrained('./三分类model')
