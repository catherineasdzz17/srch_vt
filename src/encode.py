import os
# import gdown
import torch
import towhee
# import zipfile

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from config import SENTENCE_MODEL_PATH, CLIP_MODEL_PATH


class SentenceModel:
    def __init__(self):
        # 创建模型目录
        if not os.path.exists(SENTENCE_MODEL_PATH):
            os.makedirs(SENTENCE_MODEL_PATH)
        # if not os.path.exists("./paraphrase-mpnet-base-v2.zip"):
        #     url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/paraphrase-mpnet-base-v2.zip'
        #     gdown.download(url)
        # with zipfile.ZipFile('paraphrase-mpnet-base-v2.zip', 'r') as zip_ref:
        #     zip_ref.extractall(MODEL_PATH)

        # 加载模型
        self.model = SentenceTransformer(SENTENCE_MODEL_PATH)


    def sentence_encode(self, data):
        '''
        文本编码
        '''
        embedding = self.model.encode(data)
        sentence_embeddings = normalize(embedding)  # 归一化
        return sentence_embeddings.tolist()


class Resnet50:
    def resnet50_extract_feat(self, img_path):
        '''
        提取图像特征
        '''
        # 读取图像路径，解码图像并提取特征
        feat = towhee.glob(img_path) \
            .image_decode() \
            .image_embedding.timm(model_name='resnet50') \
            .tensor_normalize() \
            .to_list()
        return feat[0]


class ChineseCLIP:
    def __init__(self, model_name="ViT-B-16"):
        print("Available models:", available_models())
        # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 加载模型和图像预处理方法
        model, preprocess = load_from_name(model_name, device=self.device, download_root=CLIP_MODEL_PATH)
        # dic = torch.load('model/cn-clip/epoch200.pt')['state_dict']
        # new_dic = collections.OrderedDict()
        # for i in dic.keys():
        #     new_dic[i[7:100]] = dic[i]
        # model.load_state_dict(new_dic)

        model.eval()
        self.model = model
        self.preprocess = preprocess


    def get_model(self):
        return self.model


    def get_preprocess(self):
        return self.preprocess


    def clip_extract_features(self, data, type):
        '''
        提取特征
        '''
        # 文本特征提取
        if type == 'text':
            text = clip.tokenize([data]).to(self.device)  # token化
            with torch.no_grad():
                features = self.model.encode_text(text)  # 提取特征
                features /= features.norm(dim=-1, keepdim=True)  # 归一化
        # 图像特征提取
        elif type == 'image':
            image = self.preprocess(data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image)  # 提取特征
                features /= features.norm(dim=-1, keepdim=True)  # 归一化
        return features