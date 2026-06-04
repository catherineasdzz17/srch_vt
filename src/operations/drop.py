import sys
sys.path.append("..")

from config import DEFAULT_TABLE
from logs import LOGGER


def do_drop(table_name, milvus_cli, mysql_cli):
    '''
    删除指定的Milvus向量集合和MySQL中的对应数据表
    '''
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        # 检查table是否存在
        if not milvus_cli.has_collection(table_name):
            return "collection is not exist"
        # 删除table
        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error with drop table: {e}")
        sys.exit(1)