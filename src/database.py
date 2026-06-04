import pymysql
from contextlib import contextmanager
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB


class DatabaseManager:
    """数据库连接管理器"""
    def __init__(self):
        self.db_config = {
            "host": MYSQL_HOST,
            "user": MYSQL_USER,
            "port": MYSQL_PORT,
            "password": MYSQL_PWD,
            "database": MYSQL_DB,
            "charset": "utf8mb4"
        }
    

    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = pymysql.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()
    

    def execute_query(self, sql, params=None, fetchone=False):
        """执行查询并返回结果"""
        with self.get_connection() as conn:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            try:
                cursor.execute(sql, params or ())
                if fetchone:
                    return cursor.fetchone()
                return cursor.fetchall()
            finally:
                cursor.close()
    

    def execute_update(self, sql, params=None):
        """执行更新操作"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params or ())
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()


# 全局实例
db_manager = DatabaseManager()