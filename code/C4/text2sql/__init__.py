"""
简化的Text2SQL框架
基于RAGFlow方案实现的Text2SQL框架
"""

__version__ = "1.0.0"
__author__ = "RAG Team"

from .knowledge_base import SimpleKnowledgeBase # 导入SimpleKnowledgeBase类
from .sql_generator import SimpleSQLGenerator # 导入SimpleSQLGenerator类
from .text2sql_agent import SimpleText2SQLAgent # 导入SimpleText2SQLAgent类

__all__ = [
    "SimpleKnowledgeBase", # 导出类
    "SimpleSQLGenerator", # 导出类
    "SimpleText2SQLAgent" # 导出类
] # 定义模块的公共接口