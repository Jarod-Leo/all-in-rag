# 这是一个使用 Pydantic 进行结构化输出解析的示例，结合了 LangChain 和 DeepSeek LLM。

from typing import List
import os
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_deepseek import ChatDeepSeek

# 初始化 LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 1. 定义数据结构
class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名") # 让 Pydantic 正确识别类型和元数据，实现结构化校验和自动文档。
    age: int = Field(description="人物年龄") # 使用 Field 添加描述，有助于生成更清晰的提示模板。
    skills: List[str] = Field(description="技能列表") 

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo) # 创建 Pydantic 输出解析器，传入定义的数据结构，以便 LLM 输出能被解析为该结构。

# 3. 创建提示模板
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"], # 定义输入变量，这里是待解析的文本。input_variables是一个列表，包含模板中所有的变量名。
    partial_variables={"format_instructions": parser.get_format_instructions()}, # 使用解析器生成的格式说明，指导 LLM 输出符合 Pydantic 模型。
)

# 打印格式指令
print("\n--- Format Instructions ---")
print(parser.get_format_instructions())
print("--------------------------\n")

# 4. 创建处理链
chain = prompt | llm | parser

# 5. 定义输入文本并执行调用链
text = "张三今年30岁，他擅长Python和Go语言。"
result = chain.invoke({"text": text})

# 6. 打印结果
print("\n--- 解析结果 ---")
print(f"结果类型: {type(result)}")
print(result)
print("--------------------\n")

print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"技能: {result.skills}")
