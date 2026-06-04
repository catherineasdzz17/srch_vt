import torch
import torch.nn as nn
from transformers import ViTModel
from config import NUM_CLASS, CLASS_MODEL_PATH


class vit_model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit = ViTModel.from_pretrained(CLASS_MODEL_PATH)  # ViT模型
        self.classification = torch.nn.Linear(768, NUM_CLASS)  # 分类器

    def forward(self, x):
        # batch_size * num_channel * width * height
        x = self.vit(x).last_hidden_state # batch_size * 197 * 768
        x = x[:, 0, :]  # batch_size * 1 * 768
        x = self.classification(x)
        x = nn.functional.softmax(x, dim=1)
        return x