# 这是一个使用函数调用（工具调用）功能的示例，展示了如何定义工具、让模型调用工具，并将工具结果返回给模型以生成最终回答。

from openai import OpenAI
import os

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 定义一个函数，用于发送消息并获取模型的响应
def send_messages(messages, tools=None):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # 让模型自主决定是否调用工具
    )
    return response.choices[0].message # 返回模型的回复消息,choices是一个列表，

# 1. 定义工具（函数）的 Schema，告诉大模型有哪些工具、每个工具的参数和用途。
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和省份，例如：杭州市, 浙江省",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

# 1. 用户提问，模型决策调用工具
messages = [{"role": "user", "content": "杭州今天天气怎么样？"}]
print(f"User> {messages[0]['content']}\n") # 打印用户问题
message = send_messages(messages, tools=tools)

# 2. 执行工具，并将结果返回模型
if message.tool_calls:
    print("--- 模型发起了工具调用 ---")
    tool_call = message.tool_calls[0] # 获取第一个工具调用请求，因为这里只有一个工具
    function_info = tool_call.function # 获取工具调用的函数信息
    print(f"工具名称: {function_info.name}") # 打印工具名称
    print(f"工具参数: {function_info.arguments}") # 打印工具参数

    # 将模型的回复（包含工具调用请求）添加到消息历史中
    messages.append(message)

    # 模拟执行工具
    tool_output = "24℃，晴朗"
    print(f"--- 执行工具并返回结果 ---")
    print(f"工具执行结果: {tool_output}\n")

    # 将工具的执行结果作为一个新的消息添加到历史中
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_output})

    # 3. 第二次调用：将工具结果返回给模型，获取最终回答
    print("--- 将工具结果返回给模型，获取最终答案 ---")
    final_message = send_messages(messages, tools=tools)
    print(f"Model> {final_message.content}")
else:
    # 如果模型没有调用工具，直接打印其回答
    print(f"Model> {message.content}")
