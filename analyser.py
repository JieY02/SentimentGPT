import torch
import logging

from torch import nn
from transformers import GPT2Tokenizer, GPT2Model
import os
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 加载模型和tokenizer
output_dir = './model'
model_path = os.path.join(output_dir, 'sentiment_emotion_model.pth')
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)


class SentimentEmotionModel(nn.Module):
    def __init__(self, gpt_model, num_sentiments, num_emotions):
        super(SentimentEmotionModel, self).__init__()
        self.gpt = gpt_model
        self.sentiment_classifier = nn.Linear(self.gpt.config.hidden_size,
                                              num_sentiments)
        self.emotion_classifier = nn.Linear(self.gpt.config.hidden_size,
                                            num_emotions)
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


# 加载预训练的GPT-2模型
model = SentimentEmotionModel(GPT2Model.from_pretrained('gpt2'), 3,
                              len(tokenizer))
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()


def analyze_text(text):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True,
        max_length=128, padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='pt')

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        sentiment_probs, emotion_probs = model(input_ids, attention_mask)

    sentiment_label = torch.argmax(sentiment_probs, dim=1).item()
    emotion_label = torch.argmax(emotion_probs, dim=1).item()

    sentiment_labels = {0: "neutral", 1: "negative", 2: "positive"}
    emotion_labels = {i: label for label, i in tokenizer.get_vocab().items() if
                      i < len(tokenizer)}

    analysis = {"text": text, "sentiment": sentiment_labels[sentiment_label],
        "sentiment_score": sentiment_probs[0][sentiment_label].item(),
        "emotion": emotion_labels[emotion_label],
        "emotion_score": emotion_probs[0][emotion_label].item()}

    return analysis


# 示例文本
# text = "I'm so happy with the service provided by the company!"
text = ""
input(text)

# 分析文本
logger.info("Analyzing text...")
analysis_result = analyze_text(text)

# 输出分析报告
report_path = os.path.join(output_dir, 'analysis_report.json')
with open(report_path, 'w') as f:
    json.dump(analysis_result, f, indent=4)

logger.info(f"Analysis report saved to {report_path}")
logger.info(json.dumps(analysis_result, indent=4))
