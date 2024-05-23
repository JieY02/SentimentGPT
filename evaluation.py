import pandas as pd
import torch
import torch.nn as nn
import logging
import os
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

# 划分数据集，缩小验证集为10%
logger.info("Splitting dataset into train and smaller validation sets...")
_, val_df = train_test_split(df, test_size=0.1, random_state=42)

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

# 加载模型
logger.info("Loading model...")
model_path = 'model/sentiment_emotion_model.pth'
model = SentimentEmotionModel(GPT2Model.from_pretrained('gpt2'), 3, len(emotion_mapping)).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 创建验证数据集
logger.info("Creating validation dataset...")
val_texts = val_df['Text'].tolist()
val_sentiments = val_df['sentiment'].tolist()
val_sentiment_scores = val_df['sentiment_score'].tolist()
val_emotions = val_df['emotion'].tolist()
val_emotion_scores = val_df['emotion_score'].tolist()

val_dataset = TweetDataset(
    val_texts, val_sentiments, val_sentiment_scores, val_emotions, val_emotion_scores, tokenizer, max_length=128
)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 评估模型
logger.info("Evaluating model...")
val_preds_sentiments, val_preds_emotions, val_labels_sentiments, val_labels_emotions = [], [], [], []
total_steps = len(val_loader)

with torch.no_grad():
    for step, batch in enumerate(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiments = batch['sentiment'].to(device)
        emotions = batch['emotion'].to(device)

        sentiment_probs, emotion_probs = model(input_ids, attention_mask)

        val_preds_sentiments.extend(torch.argmax(sentiment_probs, dim=1).cpu().numpy())
        val_labels_sentiments.extend(sentiments.cpu().numpy())
        val_preds_emotions.extend(torch.argmax(emotion_probs, dim=1).cpu().numpy())
        val_labels_emotions.extend(emotions.cpu().numpy())

        if step % 10 == 0:
            logger.info(f'Step {step}/{total_steps} - Evaluation progress')

sentiment_acc = accuracy_score(val_labels_sentiments, val_preds_sentiments)
emotion_acc = accuracy_score(val_labels_emotions, val_preds_emotions)
sentiment_precision = precision_score(val_labels_sentiments, val_preds_sentiments, average='weighted')
sentiment_recall = recall_score(val_labels_sentiments, val_preds_sentiments, average='weighted')
sentiment_f1 = f1_score(val_labels_sentiments, val_preds_sentiments, average='weighted')
emotion_precision = precision_score(val_labels_emotions, val_preds_emotions, average='weighted')
emotion_recall = recall_score(val_labels_emotions, val_preds_emotions, average='weighted')
emotion_f1 = f1_score(val_labels_emotions, val_preds_emotions, average='weighted')

logger.info(f'Sentiment Accuracy: {sentiment_acc}')
logger.info(f'Sentiment Precision: {sentiment_precision}')
logger.info(f'Sentiment Recall: {sentiment_recall}')
logger.info(f'Sentiment F1 Score: {sentiment_f1}')
logger.info(f'Emotion Accuracy: {emotion_acc}')
logger.info(f'Emotion Precision: {emotion_precision}')
logger.info(f'Emotion Recall: {emotion_recall}')
logger.info(f'Emotion F1 Score: {emotion_f1}')
