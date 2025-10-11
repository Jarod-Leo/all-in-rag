# 这是一个使用BGE-Small中文嵌入函数和Milvus向量数据库实现的Text2SQL知识库示例
# 该示例展示了如何加载SQL示例和表结构信息，创建向量存储，并进行向量检索
# 还包含一个简单的SQLite数据库查询演示

import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection


class BGESmallEmbeddingFunction:
    """BGE-Small中文嵌入函数，用于Text2SQL知识库向量化"""
    
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device) # 加载BGE-Small模型
        self.dense_dim = self.model.get_sentence_embedding_dimension() # 获取模型的向量维度
    
    def encode_text(self, texts):
        """编码文本为密集向量"""
        if isinstance(texts, str): # 如果输入是单个字符串，转换为列表
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True, # 归一化向量
            batch_size=16, # 批处理大小
            convert_to_numpy=True # 转换为NumPy数组
        )
        
        return embeddings # 返回编码后的向量
    
    @property
    def dim(self):
        """返回向量维度"""
        return self.dense_dim


class SimpleKnowledgeBase:
    """简化的知识库，使用BGE-Small进行向量检索"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri # Milvus数据库连接URI
        self.collection_name = "text2sql_knowledge_base"
        self.milvus_client = None # Milvus客户端
        self.collection = None # Milvus集合
        
        self.embedding_function = BGESmallEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"
        ) # 初始化嵌入函数
        
        self.sql_examples = [] # SQL示例列表
        self.table_schemas = [] # 表结构信息列表
        self.data_loaded = False  # 数据是否已加载标志
    
    def connect_milvus(self):
        """连接Milvus数据库"""
        connections.connect(uri=self.milvus_uri)
        self.milvus_client = MilvusClient(uri=self.milvus_uri)
        return True
    
    def create_collection(self):
        """创建Milvus集合"""
        if not self.milvus_client: # 如果没有连接Milvus，先连接
            self.connect_milvus()
        
        if self.milvus_client.has_collection(self.collection_name): # 如果集合已存在，先删除
            self.milvus_client.drop_collection(self.collection_name)
        
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="sql", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim)
        ] # 定义集合字段
        
        schema = CollectionSchema(fields, description="Text2SQL知识库") # 定义集合模式
        self.collection = Collection(name=self.collection_name, schema=schema, consistency_level="Strong") # 创建集合，使用强一致性
        
        index_params = {"index_type": "AUTOINDEX", "metric_type": "IP", "params": {}} # 定义索引参数，使用内积（IP）作为相似度度量，自动选择索引类型
        self.collection.create_index("embedding", index_params) # 创建索引，"embedding"字段用于向量索引

        return True
    
    def load_data(self):
        """加载知识库数据"""        
        data_dir = os.path.join(os.path.dirname(__file__), "data") # 数据目录，"data"参数是相对于当前脚本的路径
        
        self.load_sql_examples(data_dir) # 加载SQL示例
        self.load_table_schemas(data_dir) # 加载表结构信息
        self.vectorize_and_store() # 向量化并存储数据

        self.data_loaded = True
    
    def load_sql_examples(self, data_dir: str):
        """加载SQL示例"""
        sql_examples_path = os.path.join(data_dir, "qsql_examples.json") # SQL示例文件路径
        
        default_examples = [
            {"question": "查询所有用户信息", "sql": "SELECT * FROM users", "description": "获取用户记录", "database": "sqlite"},
            {"question": "年龄大于30的用户", "sql": "SELECT * FROM users WHERE age > 30", "description": "年龄筛选", "database": "sqlite"},
            {"question": "统计用户总数", "sql": "SELECT COUNT(*) as user_count FROM users", "description": "用户计数", "database": "sqlite"},
            {"question": "查询库存不足的产品", "sql": "SELECT * FROM products WHERE stock < 50", "description": "库存筛选", "database": "sqlite"},
            {"question": "查询用户订单信息", "sql": "SELECT u.name, p.name, o.quantity FROM orders o JOIN users u ON o.user_id = u.id JOIN products p ON o.product_id = p.id", "description": "订单详情", "database": "sqlite"},
            {"question": "按城市统计用户", "sql": "SELECT city, COUNT(*) as count FROM users GROUP BY city", "description": "城市分组", "database": "sqlite"}
        ] # 默认SQL示例
        
        if os.path.exists(sql_examples_path): # 如果文件存在，加载文件内容
            with open(sql_examples_path, 'r', encoding='utf-8') as f: # 读取文件
                self.sql_examples = json.load(f) # 加载JSON内容
        else: # 如果文件不存在，使用默认示例并保存到文件
            self.sql_examples = default_examples
            os.makedirs(data_dir, exist_ok=True) # 创建数据目录
            with open(sql_examples_path, 'w', encoding='utf-8') as f: # 保存默认示例到文件
                json.dump(self.sql_examples, f, ensure_ascii=False, indent=2)
    
    def load_table_schemas(self, data_dir: str):
        """加载表结构信息"""
        schema_path = os.path.join(data_dir, "table_schemas.json") # 表结构文件路径
        
        default_schemas = [
            {
                "table_name": "users",
                "description": "用户信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "用户ID"},
                    {"name": "name", "type": "VARCHAR", "description": "用户姓名"},
                    {"name": "age", "type": "INTEGER", "description": "用户年龄"},
                    {"name": "email", "type": "VARCHAR", "description": "邮箱地址"},
                    {"name": "city", "type": "VARCHAR", "description": "所在城市"},
                    {"name": "created_at", "type": "DATETIME", "description": "创建时间"}
                ]
            },
            {
                "table_name": "products",
                "description": "产品信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "产品ID"},
                    {"name": "product_name", "type": "VARCHAR", "description": "产品名称"},
                    {"name": "category", "type": "VARCHAR", "description": "产品类别"},
                    {"name": "price", "type": "DECIMAL", "description": "产品价格"},
                    {"name": "stock", "type": "INTEGER", "description": "库存数量"},
                    {"name": "description", "type": "TEXT", "description": "产品描述"}
                ]
            },
            {
                "table_name": "orders",
                "description": "订单信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "订单ID"},
                    {"name": "user_id", "type": "INTEGER", "description": "用户ID"},
                    {"name": "product_id", "type": "INTEGER", "description": "产品ID"},
                    {"name": "quantity", "type": "INTEGER", "description": "购买数量"},
                    {"name": "total_price", "type": "DECIMAL", "description": "总价格"},
                    {"name": "order_date", "type": "DATETIME", "description": "订单日期"}
                ]
            }
        ] # 默认表结构信息

        if os.path.exists(schema_path): # 如果文件存在，加载文件内容
            with open(schema_path, 'r', encoding='utf-8') as f: # 读取文件
                self.table_schemas = json.load(f) # 加载JSON内容
        else: # 如果文件不存在，使用默认示例并保存到文件
            self.table_schemas = default_schemas
            os.makedirs(data_dir, exist_ok=True)
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(self.table_schemas, f, ensure_ascii=False, indent=2)
    
    def vectorize_and_store(self):
        """向量化数据并存储到Milvus"""
        self.create_collection() # 创建Milvus集合
        
        all_texts = [] # 用于存储所有文本
        all_metadata = [] # 用于存储对应的元数据
        
        for example in self.sql_examples: # 处理每个SQL示例
            text = f"问题: {example['question']} SQL: {example['sql']} 描述: {example.get('description', '')}" # 构建文本
            all_texts.append(text) # 添加到文本列表
            all_metadata.append({
                "content_type": "sql_example", # 内容类型
                "question": example['question'], # 问题
                "sql": example['sql'], # SQL语句
                "description": example.get('description', ''), # 描述
                "table_name": "" # 关联表名（如果有）
            })
        
        for schema in self.table_schemas: # 处理每个表结构
            columns_desc = ", ".join([f"{col['name']} ({col['type']}): {col.get('description', '')}" 
                                    for col in schema['columns']]) # 构建列描述
            text = f"表 {schema['table_name']}: {schema['description']} 字段: {columns_desc}" # 构建文本
            all_texts.append(text) # 添加到文本列表
            all_metadata.append({
                "content_type": "table_schema", # 内容类型
                "question": "", # 问题为空
                "sql": "", # SQL语句为空
                "description": schema['description'], # 表描述
                "table_name": schema['table_name'] # 表名
            })
        
        embeddings = self.embedding_function.encode_text(all_texts) # 编码所有文本为向量
        
        insert_data = [] # 用于存储插入数据
        for i, (embedding, metadata) in enumerate(zip(embeddings, all_metadata)):
            insert_data.append([
                metadata["content_type"], # 内容类型
                metadata["question"], # 问题
                metadata["sql"], # SQL语句
                metadata["description"], # 描述
                metadata["table_name"], # 关联表名（如果有）
                embedding.tolist() # 向量数据
            ])
        
        self.collection.insert(insert_data) # 插入数据到Milvus集合
        self.collection.flush() # 刷新集合，确保数据写入
        self.collection.load() # 加载集合到内存，准备查询
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关的知识库信息"""
        if not self.data_loaded: # 如果数据未加载，先加载数据
            self.load_data()
        
        query_embedding = self.embedding_function.encode_text([query])[0]  # 编码查询文本为向量,取第一个向量因为输入是单个字符串，shape:[1, dim]
        
        search_params = {"metric_type": "IP", "params": {}} # 定义搜索参数，使用内积（IP）作为相似度度量
        results = self.collection.search( # 执行搜索
            [query_embedding.tolist()], # 查询向量，注意这里是一个二维列表，表示多个查询向量
            anns_field="embedding", # 使用的向量字段
            param=search_params, # 搜索参数
            limit=top_k, # 返回的结果数量
            output_fields=["content_type", "question", "sql", "description", "table_name"]
        )[0] # 取第一个结果，因为输入是单个查询向量, shape:[top_k, ... ]
        
        formatted_results = [] # 格式化结果
        for hit in results:
            result = {
                "score": float(hit.distance), # 相似度分数
                "content_type": hit.entity.get("content_type"), # 内容类型
                "question": hit.entity.get("question"), # 问题
                "sql": hit.entity.get("sql"), # SQL语句
                "description": hit.entity.get("description"), # 描述
                "table_name": hit.entity.get("table_name") # 关联表名
            } # 作用是把 hit.entity 里的每个字段取出来，组成一个新的字典，扁平化处理
            formatted_results.append(result) # 添加到结果列表
        
        return formatted_results
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """降级搜索方法（简单文本匹配）"""
        results = []
        query_lower = query.lower()
        
        for example in self.sql_examples:
            question_lower = example['question'].lower()
            sql_lower = example['sql'].lower()
            
            score = 0
            for word in query_lower.split():
                if word in question_lower:
                    score += 2
                if word in sql_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    "score": score,
                    "content_type": "sql_example",
                    "question": example['question'],
                    "sql": example['sql'],
                    "description": example.get('description', ''),
                    "table_name": ""
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def add_sql_example(self, question: str, sql: str, description: str = ""):
        """添加新的SQL示例"""
        new_example = {
            "question": question,
            "sql": sql,
            "description": description,
            "database": "sqlite"
        } # 新的SQL示例字典
        self.sql_examples.append(new_example) # 添加到示例列表
        
        data_dir = os.path.join(os.path.dirname(__file__), "data") # 数据目录
        sql_examples_path = os.path.join(data_dir, "qsql_examples.json") # SQL示例文件路径
        
        with open(sql_examples_path, 'w', encoding='utf-8') as f: # 保存更新后的SQL示例到文件
            json.dump(self.sql_examples, f, ensure_ascii=False, indent=2) 
        
        if self.collection and self.data_loaded: # 如果集合已创建且数据已加载，向量化并插入新的示例
            text = f"问题: {question} SQL: {sql} 描述: {description}" # 构建文本
            embedding = self.embedding_function.encode_text([text])[0] # 编码文本为向量，取第一个向量因为输入是单个字符串，shape:[1, dim]
            
            insert_data = [[
                "sql_example",
                question,
                sql,
                description,
                "",
                embedding.tolist()
            ]] # 构建插入数据，shape:[1, ...]
            
            self.collection.insert(insert_data) # 插入数据
            self.collection.flush() # 刷新集合
    
    def cleanup(self):
        """清理资源"""
        if self.collection:
            self.collection.release()
        
        if self.milvus_client and self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)


def demo():
    """简单演示"""
    # 模型测试
    embedding_function = BGESmallEmbeddingFunction() # 初始化嵌入函数
    test_texts = ["查询用户", "统计数据"] # 测试文本
    embeddings = embedding_function.encode_text(test_texts) # 编码文本为向量
    print(f"向量维度: {embeddings.shape}") # 打印向量维度
    
    # 数据库查询演示
    db_path = "demo.db" # SQLite数据库文件路径
    
    if os.path.exists(db_path): # 如果文件存在，先删除
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path) # 连接SQLite数据库
    cursor = conn.cursor() # 创建游标
    
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)") # 创建用户表，包含ID、姓名、年龄、城市字段
    
    users_data = [(1, '张三', 25, '北京'), (2, '李四', 32, '上海'), (3, '王五', 35, '深圳')] # 示例用户数据
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users_data) # 插入用户数据
    
    conn.commit() # 提交事务
    
    # 执行查询
    test_sqls = [
        ("查询所有用户", "SELECT * FROM users"),
        ("年龄大于30的用户", "SELECT * FROM users WHERE age > 30"),
        ("统计用户总数", "SELECT COUNT(*) FROM users")
    ] # 测试SQL语句
    
    for i, (question, sql) in enumerate(test_sqls, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 40)
        print(f"SQL: {sql}")
        
        cursor.execute(sql) # 执行SQL查询
        rows = cursor.fetchall() # 获取所有结果
        
        if rows: # 如果有结果，打印前两行
            print(f"返回 {len(rows)} 行数据")
            for j, row in enumerate(rows[:2], 1): # 只打印前两行
                print(f"  {j}. {row}") 
            
            if len(rows) > 2:
                print(f"  ... 还有 {len(rows) - 2} 行")
        else:
            print("无数据返回")
    
    conn.close()
    os.remove(db_path)


if __name__ == "__main__":
    demo()