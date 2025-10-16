# 文本到SQL
继上一节探讨了如何为元数据和图数据构建查询后，本节将聚焦于结构化数据领域中一个常见的应用。在数据世界中，除了向量数据库能够处理的非结构化数据，关系型数据库（如 MySQL, PostgreSQL, SQLite）同样是存储和管理结构化数据的重点。文本到SQL（Text-to-SQL）正是为了打破人与结构化数据之间的语言障碍而生。它利用大语言模型（LLM）将用户的自然语言问题，直接翻译成可以在数据库上执行的SQL查询语句。
![Text2SQL workflow](../images/4_3_1.webp)
## 一、主要挑战
- **“幻觉”问题**：LLM 可能会“想象”出数据库中不存在的表或字段，导致生成的SQL语句无效。
- **对数据库结构理解不足：** LLM需要准确理解表的结构、字段的含义以及表与表之间的关联关系，才能生成正确的 `JOIN` 和 `WHERE` 子句。
- **处理用户输入的模糊性：**用户的提问可能存在拼写错误或不规范的表达（例如，“上个月的销售冠军是谁？”），模型需要具备一定的容错和推理能力。

## 二、优化策略
1. **提供精确的数据库模式 (Schema)**：这是最基础也是最关键的一步。我们需要向LLM提供数据库中相关表的 CREATE TABLE 语句。这就像是给了LLM一张地图，让它了解数据库的结构，包括表名、列名、数据类型和外键关系。

2. **提供少量高质量的示例 (Few-shot Examples)**：在提示（Prompt）中加入一些“问题-SQL”的示例对，可以极大地提升LLM生成查询的准确性。这相当于给了LLM几个范例，让它学习如何根据相似的问题构建查询。

3. **利用RAG增强上下文：**这是更进一步的策略。我们可以像RAGFlow一样，为数据库构建一个专门的“知识库”2，其中不仅包含表的DDL（数据定义语言），还可以包含：

    - **表和字段的详细描述：**用自然语言解释每个表是做什么的，每个字段代表什么业务含义。
    - **同义词和业务术语：**例如，将用户的“花费”映射到数据库的 `cost` 字段。
    - **复杂的查询示例**：提供一些包含 `JOIN、GROUP BY` 或子查询的复杂问答对。 当用户提问时，系统首先从这个知识库中检索最相关的信息（如相关的表结构、字段描述、相似的Q&A），然后将这些信息和用户的问题一起组合成一个内容更丰富的提示，交给LLM生成最终的SQL查询。这种方式极大地降低了“幻觉”的风险，提高了查询的准确度。

4. **错误修正与反思 (Error Correction and Reflection)**：在生成SQL后，系统会尝试执行它。如果数据库返回错误，可以将错误信息反馈给LLM，让它“反思”并修正SQL语句，然后重试。这个迭代过程可以显著提高查询的成功率。

## 三、实现一个简单的Tesx2SQL框架
本节基于RAGFlow方案实现了一个简单的Text2SQL框架。该框架使用Milvus向量数据库作为知识库，BGE-M3模型进行语义检索，DeepSeek作为大语言模型，专门针对SQLite数据库进行了优化。
![Text2SQL框架](../images/4_3_2.webp)

### 3.1 知识库模块（`knowledge_base.py`）
```python
class SimpleKnowledgeBase:
    """知识库"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri
        self.client = MilvusClient(uri=milvus_uri)
        self.embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.collection_name = "text2sql_kb"
        self._setup_collection()
```
**设计思想：**
1. **统一知识管理：**将DDL定义、Q-SQL示例和表描述三种类型的知识统一存储在一个Milvus集合中，通过 `type` 字段区分。

2. **语义检索能力：**使用BGE-M3模型进行向量化，支持中英文混合的语义相似度搜索。

```python
def _setup_collection(self):
    """设置集合"""
    # 定义字段
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),  # ddl, qsql, description
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim["dense"])
    ]
```
数据加载策略：
```python
def load_data(self):
    """加载所有知识库数据"""
    # 加载DDL数据 - 表结构定义
    # 加载Q->SQL数据 - 问答示例
    # 加载描述数据 - 表和字段的业务描述
```
框架支持三种类型的知识：

- **DDL知识：**表的结构定义，包括字段类型、约束等
- **Q-SQL知识：**历史问答对，为新问题提供参考模式
- **描述知识：**表和字段的业务含义，帮助理解数据语义
检索机制
```python
def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """搜索相关内容"""
    query_embeddings = self.embedding_function([query])
    
    search_results = self.client.search(
        collection_name=self.collection_name,
        data=query_embeddings["dense"],
        anns_field="dense_vector",
        search_params={"metric_type": "IP"},  # 内积相似度
        limit=top_k,
        output_fields=["content", "type"]
    )
```
### 3.2 SQL生成模块（sql_generator.py）
SQL生成模块负责将自然语言问题转换为SQL查询语句，并具备错误修复能力。
```python
class SimpleSQLGenerator:
    """简化的SQL生成器"""
    
    def __init__(self, api_key: str = None):
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,  # 确保结果的确定性
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY")
        )
```
**SQL生成策略：**
```python
def generate_sql(self, user_query: str, knowledge_results: List[Dict[str, Any]]) -> str:
    """生成SQL语句"""
    # 构建上下文
    context = self._build_context(knowledge_results)
    
    # 构建提示
    prompt = f"""你是一个SQL专家。请根据以下信息将用户问题转换为SQL查询语句。

数据库信息：
{context}

用户问题：{user_query}

要求：
1. 只返回SQL语句，不要包含任何解释
2. 确保SQL语法正确
3. 使用上下文中提供的表名和字段名
4. 如果需要JOIN，请根据表结构进行合理关联

SQL语句："""
```
**关键设计原则：**

1. **上下文驱动：**通过知识库检索结果构建丰富的上下文信息
2. **结构化提示：**明确的任务要求和格式约束
3. **确定性输出：**设置temperature=0确保相同输入产生相同输出

**错误修复机制：**
```python
def fix_sql(self, original_sql: str, error_message: str, knowledge_results: List[Dict[str, Any]]) -> str:
    """修复SQL语句"""
    context = self._build_context(knowledge_results)
    
    prompt = f"""请修复以下SQL语句的错误。

数据库信息：
{context}

原始SQL：
{original_sql}

错误信息：
{error_message}

请返回修复后的SQL语句（只返回SQL，不要解释）："""
```
**上下文构建策略：**
```python
def _build_context(self, knowledge_results: List[Dict[str, Any]]) -> str:
    """构建上下文信息"""
    # 按类型分组
    ddl_info = []        # 表结构信息
    qsql_examples = []   # 查询示例
    descriptions = []    # 表描述信息
    
    # 分层次组织信息：结构 → 描述 → 示例
    if ddl_info:
        context += "=== 表结构信息 ===\n"
    if descriptions:
        context += "=== 表和字段描述 ===\n"
    if qsql_examples:
        context += "=== 查询示例 ===\n"
```
### 3.3 代理模块 (`text2sql_agent.py`)
代理模块是整个框架的控制中心，协调知识库检索、SQL生成和执行的完整流程。

```python
class SimpleText2SQLAgent:
    """Text2SQL代理"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530", api_key: str = None):
        self.knowledge_base = SimpleKnowledgeBase(milvus_uri)
        self.sql_generator = SimpleSQLGenerator(api_key)
        
        # 配置参数
        self.max_retry_count = 3      # 最大重试次数
        self.top_k_retrieval = 5      # 检索数量
        self.max_result_rows = 100    # 结果行数限制
```
**主要查询流程：**
```python
def query(self, user_question: str) -> Dict[str, Any]:
    """执行Text2SQL查询"""
    # 1. 从知识库检索相关信息
    knowledge_results = self.knowledge_base.search(user_question, self.top_k_retrieval)
    
    # 2. 生成SQL语句
    sql = self.sql_generator.generate_sql(user_question, knowledge_results)
    
    # 3. 执行SQL（带重试机制）
    retry_count = 0
    while retry_count < self.max_retry_count:
        success, result = self._execute_sql(sql)
        
        if success:
            return {"success": True, "sql": sql, "results": result}
        else:
            # 尝试修复SQL
            sql = self.sql_generator.fix_sql(sql, result, knowledge_results)
            retry_count += 1
```
**安全执行策略：**
```python
def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
    """执行SQL语句"""
    # 添加LIMIT限制，防止大量数据返回
    # 检查是否为SELECT语句且不包含LIMIT子句
    if sql.strip().upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
        # 移除末尾分号后添加LIMIT限制
        sql = f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"
    
    # 结构化结果返回
    # 检查是否为SELECT查询语句
    if sql.strip().upper().startswith('SELECT'):
        # 获取查询结果的列名列表
        columns = [desc[0] for desc in cursor.description]
        # 获取所有查询结果行
        rows = cursor.fetchall()
        
        # 构建结构化结果列表
        results = []
        # 遍历每一行数据
        for row in rows:
            # 创建字典存储单行数据
            result_row = {}
            # 遍历行中的每个值和对应的列名
            for i, value in enumerate(row):
                # 以列名为键，值为数据构建字典
                result_row[columns[i]] = value
            # 将单行字典添加到结果列表
            results.append(result_row)
        
        # 返回执行成功标志和结构化结果
        return True, {"columns": columns, "rows": results, "count"
```