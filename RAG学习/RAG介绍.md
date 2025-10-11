# RAG介绍及环境配置
RAG（Retrieval-Augmented Generation）是一种生成式增强的检索技术，它**融合信息检索与文本生成**的技术范式。其核心逻辑是：在大型语言模型（LLM）生成文本前，先通过检索机制从外部知识库中动态获取相关信息，并将检索结果融入生成过程，从而提升输出的准确性和时效性。
## 为什么需要RAG
由于大语言模型（LLM）本质上是基于统计的神经网络模型，其token输出是按概率选择的带有一定噪声和随机性的，因此模型输出多少会存在一些幻觉，这对于要求十分严谨的场景中比如：医疗、法律、金融领域是不可接受的。

另一方面由于给模型训练的预料不可能包含各个行业的所有资料，语料库也没有实时更新，对于一些专业知识LLM对近期发展的一些技术可能会回答错误。我们迫切需要一个挂载的外部知识库来更新信息，知识溯源，因此考虑这些种种因此，RAG技术应运而生。
| 问题               | RAG 的解决方案                                 |
|--------------------|-------------------------------------------------|
| 静态知识局限       | 实时检索外部知识库，支持动态更新                 |
| 幻觉 (Hallucination) | 基于检索内容生成，错误率降低                     |
| 领域专业性不足     | 引入领域特定知识库（如医疗/法律）                |
| 数据隐私风险       | 本地化部署知识库，避免敏感数据泄露               |

## RAG组件构成及执行流程
最简单或者原始的RAG一般包含三大组件分别是
- **索引（Indexing）** 📑：将非结构化文档（PDF/Word等）分割为片段，通过嵌入模型转换为向量数据。
- **检索（Retrieval）** 🔍️：基于查询语义，从向量数据库召回最相关的文档片段（Context）。
- **生成（Generation）** ✨：将检索结果作为上下文输入LLM，生成自然语言响应。

具体执行过程是，先将文档读取进行向量嵌入存入向量数据库，然后给向量数据库中的向量构建索引，当用户提问时，检索器会根据设置的检索算法匹配数据库中top k个最符合的向量，然后将向量对应的原始文档传入LLM进行答案生成。
![RAG架构运行图](../docs/chapter1/images/1_1.svg)

## RAG开发的环境配置
### Github Codespace环境配置
**1. Deepseek API申请**
- 访问 Deepseek 开放平台 打开浏览器，访问 Deepseek 开放平台。
![deepseek](../docs/chapter1/images/1_2_1.webp)

- 登录或注册账号 如果你已有账号，请直接登录。如果没有，请点击页面上的注册按钮，使用邮箱或手机号完成注册。

- 创建新的 API 密钥 登录成功后，在页面左侧的导航栏中找到并点击 API Keys。在 API 管理页面，点击 创建 API key 按钮。输入一个跟其他api key不重复的名称后点击创建
![api-key创建](../docs/chapter1/images/1_2_3.webp)
- 保存 API Key 系统会为你生成一个新的 API 密钥。请立即复制并将其保存在一个安全的地方。

**2. 创建codespace空间**
在GitHub repository下按'.'即可进入codespace空间
**3. python环境配置**
在终端界面做如下操作
- a 更新系统软件包
```bash
sudo apt update
sudo apt upgrade -y
```
- b 安装Miniconda
    - 按 Enter 阅读许可协议
    - 输入 yes 同意协议
    - 安装路径提示时直接按 Enter（使用默认路径 /home/ubuntu/miniconda3）
    - 是否初始化Miniconda：输入 yes 将Miniconda添加到您的PATH环境变量中。
```bash
source ~/.bashrc
conda --version
```
**4. DEEPSEEK_API_KEY配置**
- a. 使用 vim 编辑器打开你的 shell 配置文件。
```bash
vim ~/.bashrc
```
- b. 输入 i 进入编辑模式，在文件末尾添加以下行，将 [你的 Deepseek API 密钥] 替换为你自己的密钥：
```bash
export DEEPSEEK_API_KEY=[你的 Deepseek API 密钥]
```
- c. 保存并退出 在 vim 中，按 Esc 键进入命令模式，然后输入 :wq 并按 Enter 键保存文件并退出。
- d. 使配置生效 执行以下命令来立即加载更新后的配置，让环境变量生效：
```bash
source ~/.bashrc
```
**5. 创建并激活虚拟环境**
- a. 创建虚拟环境
```bash
conda create --name all-in-rag python=3.12.7
Copy to clipboardErrorCopied
```
出现选项直接回车即可。

- b. 激活虚拟环境

使用以下命令激活虚拟环境：
```bash
conda activate all-in-rag
Copy to clipboardErrorCopied
```
- c. 依赖安装 如果严格安装上述流程当前应该在项目根目录，进入code目录安装依赖库
```bash
cd code
pip install -r requirements.txt
```
### 本地Windows环境配置