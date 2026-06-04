import sys
sys.path.append("..")

from config import DEFAULT_TABLE
from logs import LOGGER


def do_count(table_name, milvus_cli):
    '''
    统计Milvus中某个table的向量数量
    '''
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        # 检查table是否存在
        if not milvus_cli.has_collection(table_name):
            return None
        # 统计数量
        num = milvus_cli.count(table_name)
        return num
    except Exception as e:
        LOGGER.error( f"Error with count table {e}")
        sys.exit(1)