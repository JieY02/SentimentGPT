import pandas as pd
import torch
import torch.nn as nn
import logging
import os
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 加载数据集
logger.info("Loading dataset...")
data_path = 'data/sentiment-emotion-labelled_Dell_tweets.csv'
df = pd.read_csv(data_path)

# 提取需要的列
logger.info("Extracting required columns...")
df = df[['Text', 'sentiment', 'sentiment_score', 'emotion', 'emotion_score']]

# 数据预处理
logger.info("Preprocessing data...")
def preprocess_text(text):
    return text.replace(':)', 'happy').replace(':(', 'sad')

df['Text'] = df['Text'].apply(preprocess_text)

# 编码标签
logger.info("Encoding labels...")
sentiment_mapping = {"neutral": 0, "negative": 1, "positive": 2}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

emotion_mapping = {emotion: i for i, emotion in enumerate(df['emotion'].unique())}
df['emotion'] = df['emotion'].map(emotion_mapping)

# 自定义数据集
logger.info("Creating custom dataset...")
class TweetDataset(Dataset):
    def __init__(self, texts, sentiments, sentiment_scores, emotions, emotion_scores, tokenizer, max_length):
        self.texts = texts
        self.sentiments = sentiments
        self.sentiment_scores = sentiment_scores
        self.emotions = emotions
        self.emotion_scores = emotion_scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        sentiment = self.sentiments[index]
        sentiment_score = self.sentiment_scores[index]
        emotion = self.emotions[index]
        emotion_score = self.emotion_scores[index]

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length, return_token_type_ids=False,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'sentiment_score': torch.tensor(sentiment_score, dtype=torch.float),
            'emotion': torch.tensor(emotion, dtype=torch.long),
            'emotion_score': torch.tensor(emotion_score, dtype=torch.float)
        }

# 加载预训练的GPT-2模型和tokenizer
logger.info("Loading pre-trained GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# 设置模型的pad_token_id
logger.info("Setting pad_token_id...")
tokenizer.pad_token = tokenizer.eos_token

# 定义自定义分类模型
logger.info("Defining custom classification model...")
class SentimentEmotionModel(nn.Module):
    def __init__(self, gpt_model, num_sentiments, num_emotions):
        super(SentimentEmotionModel, self).__init__()
        self.gpt = gpt_model
        self.sentiment_classifier = nn.Linear(self.gpt.config.hidden_size, num_sentiments)
        self.emotion_classifier = nn.Linear(self.gpt.config.hidden_size, num_emotions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]

        sentiment_logits = self.sentiment_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)

        sentiment_probs = self.softmax(sentiment_logits)
        emotion_probs = self.softmax(emotion_logits)

        return sentiment_probs, emotion_probs

# 划分数据集
logger.info("Splitting dataset into train and validation sets...")
train_texts, val_texts, train_sentiments, val_sentiments, train_sentiment_scores, val_sentiment_scores, train_emotions, val_emotions, train_emotion_scores, val_emotion_scores = train_test_split(
    df['Text'], df['sentiment'], df['sentiment_score'], df['emotion'], df['emotion_score'], test_size=0.2, random_state=42
)

# 创建数据集
logger.info("Creating datasets...")
train_dataset = TweetDataset(
    train_texts.tolist(), train_sentiments.tolist(), train_sentiment_scores.tolist(), train_emotions.tolist(),
    train_emotion_scores.tolist(), tokenizer, max_length=128
)

val_dataset = TweetDataset(
    val_texts.tolist(), val_sentiments.tolist(), val_sentiment_scores.tolist(), val_emotions.tolist(),
    val_emotion_scores.tolist(), tokenizer, max_length=128
)

# 创建数据加载器
logger.info("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 初始化模型
logger.info("Initializing model...")
num_sentiments = 3  # neutral, negative, positive
num_emotions = len(emotion_mapping)  # 根据数据集中的情绪种类数量来确定

model = SentimentEmotionModel(model, num_sentiments, num_emotions).to(device)

# 定义优化器和损失函数
logger.info("Defining optimizer and criterion...")
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练模型
logger.info("Starting training...")
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiments = batch['sentiment'].to(device)
        emotions = batch['emotion'].to(device)

        optimizer.zero_grad()
        sentiment_probs, emotion_probs = model(input_ids, attention_mask)

        sentiment_loss = criterion(sentiment_probs, sentiments)
        emotion_loss = criterion(emotion_probs, emotions)

        loss = sentiment_loss + emotion_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

# 验证模型
logger.info("Starting evaluation...")
def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    sentiment_preds, sentiment_true = [], []
    emotion_preds, emotion_true = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiments = batch['sentiment'].to(device)
            emotions = batch['emotion'].to(device)

            sentiment_probs, emotion_probs = model(input_ids, attention_mask)

            sentiment_loss = criterion(sentiment_probs, sentiments)
            emotion_loss = criterion(emotion_probs, emotions)

            loss = sentiment_loss + emotion_loss
            total_loss += loss.item()

            sentiment_preds.extend(torch.argmax(sentiment_probs, dim=1).cpu().numpy())
            sentiment_true.extend(sentiments.cpu().numpy())
            emotion_preds.extend(torch.argmax(emotion_probs, dim=1).cpu().numpy())
            emotion_true.extend(emotions.cpu().numpy())

    sentiment_acc = accuracy_score(sentiment_true, sentiment_preds)
    emotion_acc = accuracy_score(emotion_true, emotion_preds)

    return total_loss / len(data_loader), sentiment_acc, emotion_acc

# 训练与评估
num_epochs = 3
for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, sentiment_acc, emotion_acc = eval_model(model, val_loader, criterion, device)
    logger.info(f'Train loss: {train_loss}')
    logger.info(f'Val loss: {val_loss}')
    logger.info(f'Sentiment accuracy: {sentiment_acc}')
    logger.info(f'Emotion accuracy: {emotion_acc}')

# 最终模型评估
logger.info("Evaluating final model...")
model.eval()
val_preds_sentiments, val_preds_emotions, val_labels_sentiments, val_labels_emotions = [], [], [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiments = batch['sentiment'].to(device)
        emotions = batch['emotion'].to(device)

        sentiment_probs, emotion_probs = model(input_ids, attention_mask)

        val_preds_sentiments.extend(torch.argmax(sentiment_probs, dim=1).cpu().numpy())
        val_labels_sentiments.extend(sentiments.cpu().numpy())
        val_preds_emotions.extend(torch.argmax(emotion_probs, dim=1).cpu().numpy())
        val_labels_emotions.extend(emotions.cpu().numpy())

sentiment_acc = accuracy_score(val_labels_sentiments, val_preds_sentiments)
emotion_acc = accuracy_score(val_labels_emotions, val_preds_emotions)
sentiment_precision = precision_score(val_labels_sentiments, val_preds_sentiments, average='weighted')
sentiment_recall = recall_score(val_labels_sentiments, val_preds_sentiments, average='weighted')
sentiment_f1 = f1_score(val_labels_sentiments, val_preds_sentiments, average='weighted')
emotion_precision = precision_score(val_labels_emotions, val_preds_emotions, average='weighted')
emotion_recall = recall_score(val_labels_emotions, val_preds_emotions, average='weighted')
emotion_f1 = f1_score(val_labels_emotions, val_preds_emotions, average='weighted')

logger.info(f'Final Sentiment Accuracy: {sentiment_acc}')
logger.info(f'Final Sentiment Precision: {sentiment_precision}')
logger.info(f'Final Sentiment Recall: {sentiment_recall}')
logger.info(f'Final Sentiment F1 Score: {sentiment_f1}')
logger.info(f'Final Emotion Accuracy: {emotion_acc}')
logger.info(f'Final Emotion Precision: {emotion_precision}')
logger.info(f'Final Emotion Recall: {emotion_recall}')
logger.info(f'Final Emotion F1 Score: {emotion_f1}')

# 保存模型
output_dir = './model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger.info("Saving model...")
model_path = os.path.join(output_dir, 'sentiment_emotion_model.pth')
torch.save(model.state_dict(), model_path)
tokenizer.save_pretrained(output_dir)
logger.info(f'Model saved to {model_path}')
