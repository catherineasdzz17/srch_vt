import os

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
INDEX_FILE_SIZE = int(os.getenv("INDEX_FILE_SIZE", "1024"))
METRIC_TYPE = os.getenv("METRIC_TYPE", "IP")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "test_table")
TOP_K = int(os.getenv("TOP_K", "9"))

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "117.50.220.157")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "qqzcy@1014")
MYSQL_DB = os.getenv("MYSQL_DB", "spider_info")

############### Number of log files ###############
LOGS_NUM = int(os.getenv("logs_num", "0"))

############## Model path #################
SENTENCE_MODEL_PATH = 'model/sentence-transformers'
CLIP_MODEL_PATH = 'model/cn-clip/'
SAM_TYPE = 'default'
SAM_CHECKPOINT = 'model/segment_anything/sam_vit_h_4b8939.pth'
TFIDF_PATH = 'data'
CLASS_MODEL_PATH = 'model/ViT'
CLASS_CHECKPOINT = 'model/ViT/rgb_class5_1k_epoch100_1000.pkl'
AUDIO_MODEL_PATH = 'model/whisper-large-v3-turbo'

############### Data Path ###############
UPLOAD_IMAGE_PATH = os.getenv("UPLOAD_PATH", "tmp/search_images")
SEGMENT_SAVE_PATH = os.getenv("SAVE_PATH", "tmp/segment_images")
UPLOAD_VIDEO_PATH = os.getenv("VIDEO_PATH", "tmp/search_videos")
UPLOAD_AUDIO_PATH = os.getenv("AUDIO_PATH", "tmp/search_audios")

NUM_CLASS = 5