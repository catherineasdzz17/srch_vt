import pymysql
import sys
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB
from logs import LOGGER


class MySQLHelper():
    def __init__(self):
        # 建立与MySQL的初始连接，并创建游标
        self.conn = pymysql.connect(
            host=MYSQL_HOST, 
            user=MYSQL_USER, 
            port=MYSQL_PORT, 
            password=MYSQL_PWD, 
            database=MYSQL_DB, 
            local_infile=True
        )
        self.cursor = self.conn.cursor()


    def test_connection(self):
        """
        检测数据库连接是否有效，如果断开则重新连接
        """
        try:
            self.conn.ping()
        except Exception:
            self.conn = pymysql.connect(
                host=MYSQL_HOST, 
                user=MYSQL_USER, 
                port=MYSQL_PORT, 
                password=MYSQL_PWD, 
                database=MYSQL_DB, 
                local_infile=True
            )
            self.cursor = self.conn.cursor()


    def create_mysql_table(self, table_name):
        """
        如果不存在，则创建存储文本数据的表
        表结构：
          milvus_id : TEXT
          title     : TEXT (utf8mb4)
          text      : TEXT (utf8mb4)
        """
        self.test_connection()
        sql = (
            "create table if not exists " + table_name + 
            "(milvus_id TEXT, title TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci, "
            "text TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci);"
        )
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def load_data_to_mysql(self, table_name, data):
        """
        批量插入数据到指定表
        """
        self.test_connection()
        sql = "insert into " + table_name + " (milvus_id, title, text) values (%s, %s, %s);"
        try:
            self.cursor.executemany(sql, data)
            self.conn.commit()
            LOGGER.debug(f"MYSQL loads data to table: {table_name} successfully")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def search_by_milvus_ids(self, ids, table_name):
        """
        根据milvus_id列表查询完整记录，并保持输入顺序
        返回: (id列表, title列表, text列表)
        """
        # Get the img_path according to the milvus ids
        self.test_connection()
        str_ids = str(ids).replace('[', '').replace(']', '')
        sql = "select * from " + table_name + " where milvus_id in (" + str_ids + ") order by field (milvus_id," + str_ids + ");"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            results_id = [res[0] for res in results]
            results_title = [res[1] for res in results]
            results_text = [res[2] for res in results]
            LOGGER.debug("MYSQL search by milvus id.")
            return results_id, results_title, results_text
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def create_mysql_img_table(self, table_name):
        """
        创建仅存储图片路径的表
        表结构：
          milvus_id : TEXT
          image_path: TEXT
        """
        self.test_connection()
        sql = "create table if not exists " + table_name + "(milvus_id TEXT, image_path TEXT);"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def load_data_to_img_mysql(self, table_name, data):
        """
        批量插入数据到图片表
        """
        self.test_connection()
        sql = "insert into " + table_name + " (milvus_id, image_path) values (%s, %s);"
        try:
            self.cursor.executemany(sql, data)
            self.conn.commit()
            LOGGER.debug(f"MYSQL loads data to table: {table_name} successfully")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def search_by_milvus_img_ids(self, ids, table_name):
        """
        根据milvus_id列表查询图片路径，并保持输入顺序
        返回: image_path列表
        """
        self.test_connection()
        str_ids = str(ids).replace('[', '').replace(']', '')
        sql = "select image_path from " + table_name + " where milvus_id in (" + str_ids + ") order by field (milvus_id," + str_ids + ");"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            results = [res[0] for res in results]
            LOGGER.debug("MYSQL search by milvus id.")
            return results
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def delete_table(self, table_name):
        """
        删除整个表（如果存在）
        """
        self.test_connection()
        sql = "drop table if exists " + table_name + ";"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL delete table: {table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def delete_all_data(self, table_name):
        """
        删除表中所有数据，但保留表结构
        """
        self.test_connection()
        sql = 'delete from ' + table_name + ';'
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            LOGGER.debug(f"MYSQL delete all data in table: {table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)


    def count_table(self, table_name):
        """
        统计表中数据的数量
        """
        self.test_connection()
        sql = "select count(milvus_id) from " + table_name + ";"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            LOGGER.debug(f"MYSQL count table: {table_name}")
            return results[0][0]
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)