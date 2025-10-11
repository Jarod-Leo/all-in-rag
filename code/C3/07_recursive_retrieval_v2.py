# 这是一个使用两步式检索从多个Excel工作表中查询数据的示例。
# 第一步在摘要索引中找到相关的工作表，第二步在内容索引中根据工作表名称过滤并检索具体内容。

import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv() # 从 .env 文件加载环境变量

# 配置模型
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1. 加载和预处理数据
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

summary_docs = []
content_docs = []

print("开始加载和处理Excel文件...")
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name) # 读取每个工作表为DataFrame
    
    # 数据清洗
    if '评分人数' in df.columns:
        # .str.xxx() 是 pandas 专门为 Series 设计的批量字符串处理接口，适用于一整列数据。
        # 直接 xxx() 只适用于单个字符串。
        # 所以要用 .str.replace 和 .str.strip。
        df['评分人数'] = df['评分人数'].astype(str).str.replace('人评价', '').str.strip() # 去掉'人评价'后缀,.astype(str).str表示转换为字符串并进行字符串操作，.replace().str表示替换字符串
        df['评分人数'] = pd.to_numeric(df['评分人数'], errors='coerce').fillna(0).astype(int) # 转换为整数，无法转换的设置为0

    # 创建摘要文档 (用于路由)
    year = sheet_name.replace('年份_', '') # 提取年份
    summary_text = f"这个表格包含了年份为 {year} 的电影信息，包括电影名称、导演、评分、评分人数等。" # 摘要内容自定义
    summary_doc = Document(
        text=summary_text,
        metadata={"sheet_name": sheet_name}
    ) # 保留工作表名称作为元数据，方便后续过滤
    summary_docs.append(summary_doc)
    
    # 创建内容文档 (用于最终问答)
    content_text = df.to_string(index=False) # 将DataFrame转换为字符串
    content_doc = Document(
        text=content_text,
        metadata={"sheet_name": sheet_name}
    ) # 保留工作表名称作为元数据，方便后续过滤
    content_docs.append(content_doc) 

print("数据加载和处理完成。\n")

# 2. 构建向量索引
# 使用默认的内存SimpleVectorStore，它支持元数据过滤

# 2.1 为摘要创建索引
summary_index = VectorStoreIndex(summary_docs)

# 2.2 为内容创建索引
content_index = VectorStoreIndex(content_docs)

print("摘要索引和内容索引构建完成。\n")

# 3. 定义两步式查询逻辑
def query_safe_recursive(query_str):
    print(f"--- 开始执行查询 ---")
    print(f"查询: {query_str}")
    
    # 第一步：路由 - 在摘要索引中找到最相关的表格
    print("\n第一步：在摘要索引中进行路由...")
    summary_retriever = VectorIndexRetriever(index=summary_index, similarity_top_k=1) # 摘要向量表索引检索器
    retrieved_nodes = summary_retriever.retrieve(query_str) # 返回的是 IndexNode 列表
    
    if not retrieved_nodes:
        return "抱歉，未能找到相关的电影年份信息。"
    
    # 获取匹配到的工作表名称
    matched_sheet_name = retrieved_nodes[0].node.metadata['sheet_name']
    print(f"路由结果：匹配到工作表 -> {matched_sheet_name}")
    
    # 第二步：检索 - 在内容索引中根据工作表名称过滤并检索具体内容
    print("\n第二步：在内容索引中检索具体信息...")
    content_retriever = VectorIndexRetriever(
        index=content_index,
        similarity_top_k=1, # 通常只返回最匹配的整个表格即可
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="sheet_name", value=matched_sheet_name)]
        )
    ) # 内容向量表索引检索器，添加元数据过滤器，返回指定工作表的内容
    
    # 创建查询引擎并执行查询
    query_engine = RetrieverQueryEngine.from_args(content_retriever) # 创建查询引擎
    response = query_engine.query(query_str) # 执行查询，engine会调用retriever和llm，既包含检索也包含生成
    
    print("--- 查询执行结束 ---\n")
    return response

# 4. 执行查询
query = "1994年评分人数最少的电影是哪一部？"
response = query_safe_recursive(query)

print(f"最终回答: {response}")
