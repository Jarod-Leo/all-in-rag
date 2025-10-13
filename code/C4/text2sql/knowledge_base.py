# 这是一个简单的知识库实现，使用Milvus作为向量数据库，BGEM3作为嵌入模型

import json
import os
from typing import List, Dict, Any
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class SimpleKnowledgeBase:
    """知识库"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri
        self.client = MilvusClient(uri=milvus_uri) # 连接Milvus服务器
        self.embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu") # 使用BGEM3嵌入模型
        self.collection_name = "text2sql_kb" # 集合名称
        self._setup_collection() # 设置集合
    
    def _setup_collection(self): 
        """设置集合"""
        if self.client.has_collection(self.collection_name): # 如果集合已存在，先删除
            self.client.drop_collection(self.collection_name)
        
        # 定义字段
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),  # ddl, qsql, description
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim["dense"])
        ]
        
        schema = CollectionSchema(fields, description="Text2SQL知识库")
        
        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Strong" # 强一致性
        )
        
        # 创建索引
        index_params = self.client.prepare_index_params() # 准备索引参数
        index_params.add_index(
            field_name="dense_vector", # 向量字段，必须与上面定义的字段名一致
            index_type="AUTOINDEX", # 自动选择索引类型
            metric_type="IP"
        )
        
        self.client.create_index(  # 创建索引
            collection_name=self.collection_name,
            index_params=index_params 
        )
    
    def load_data(self):
        """加载所有知识库数据"""
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        # 加载DDL数据
        ddl_path = os.path.join(data_dir, "ddl_examples.json")
        if os.path.exists(ddl_path):
            with open(ddl_path, 'r', encoding='utf-8') as f: # 读取DDL数据文件
                ddl_data = json.load(f) # 解析JSON数据
            self._add_ddl_data(ddl_data) # 添加DDL数据
        
        # 加载Q->SQL数据
        qsql_path = os.path.join(data_dir, "qsql_examples.json")
        if os.path.exists(qsql_path):
            with open(qsql_path, 'r', encoding='utf-8') as f:
                qsql_data = json.load(f)
            self._add_qsql_data(qsql_data)
        
        # 加载描述数据
        desc_path = os.path.join(data_dir, "db_descriptions.json")
        if os.path.exists(desc_path):
            with open(desc_path, 'r', encoding='utf-8') as f:
                desc_data = json.load(f)
            self._add_description_data(desc_data)
        
        # 加载集合到内存
        self.client.load_collection(collection_name=self.collection_name)
        print("知识库数据加载完成")
    
    def _add_ddl_data(self, data: List[Dict]):
        """添加DDL数据"""
        contents = [] # 记录内容
        types = [] # 记录类型
        
        for item in data:
            content = f"表名: {item.get('table_name', '')}\n" # 构建内容字符串
            content += f"DDL: {item.get('ddl_statement', '')}\n" # 构建DDL语句字符串
            content += f"描述: {item.get('description', '')}" # 构建描述字符串

            contents.append(content) # 添加内容
            types.append("ddl") # 添加数据类型
        
        self._insert_data(contents, types) # 插入数据
    
    def _add_qsql_data(self, data: List[Dict]):
        """添加Q->SQL数据"""
        contents = [] # 记录内容
        types = [] # 记录类型
        
        for item in data:
            content = f"问题: {item.get('question', '')}\n" # 构建问题字符串
            content += f"SQL: {item.get('sql', '')}" # 构建SQL字符串

            contents.append(content) # 添加内容
            types.append("qsql") # 添加数据类型

        self._insert_data(contents, types) # 插入数据
    
    def _add_description_data(self, data: List[Dict]):
        """添加描述数据"""
        contents = [] # 记录内容
        types = [] # 记录类型
        
        for item in data:
            content = f"表名: {item.get('table_name', '')}\n" # 构建内容字符串
            content += f"表描述: {item.get('table_description', '')}\n" # 构建表描述字符串
            
            columns = item.get('columns', []) # 获取字段信息
            if columns:
                content += "字段信息:\n" # 构建字段信息字符串
                for col in columns:
                    content += f"  - {col.get('name', '')}: {col.get('description', '')} ({col.get('type', '')})\n" # 构建每个字段的描述

            contents.append(content) # 添加内容
            types.append("description") # 添加数据类型

        self._insert_data(contents, types) # 插入数据
    
    def _insert_data(self, contents: List[str], types: List[str]):
        """插入数据"""
        if not contents: # 如果没有内容，直接返回
            return
        
        # 生成嵌入
        embeddings = self.embedding_function(contents) 
        
        # 构建插入数据，每一行是一个字典
        data_to_insert = [] # 记录插入数据
        for i in range(len(contents)): 
            data_to_insert.append({
                "content": contents[i],
                "type": types[i],
                "dense_vector": embeddings["dense"][i]
            }) # 添加插入数据
        
        # 插入数据
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        ) # 执行插入，插入到milvus集合中，返回插入结果
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关内容"""
        self.client.load_collection(collection_name=self.collection_name)
            
        query_embeddings = self.embedding_function([query]) # 生成查询嵌入,[query]表示输入是一个列表
        
        search_results = self.client.search( # 执行搜索
            collection_name=self.collection_name, # 集合名称
            data=query_embeddings["dense"], # 密集向量嵌入
            anns_field="dense_vector", # 密集向量搜索
            search_params={"metric_type": "IP"}, # 搜索度量内积相似度
            limit=top_k, # 返回前top_k个结果
            output_fields=["content", "type"] # 返回内容和类型字段
        )
        
        results = []
        for hit in search_results[0]: # 只处理第一个查询的结果
            results.append({
                "content": hit["entity"]["content"], # "entity"包含了返回的字段,"content"字段
                "type": hit["entity"]["type"], 
                "score": hit["distance"]
            }) # 添加结果,筛选出内容和类型字段，以及相似度分数
        
        return results
    
    def cleanup(self):
        """清理资源"""
        try:
            self.client.drop_collection(self.collection_name)
        except:
            pass 