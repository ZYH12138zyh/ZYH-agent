#rag核心服务（在线版本）
# LangChain文档对象，向量检索返回的文档数据结构
#这是一套基于 **LangChain + Chroma 向量库 + 阿里云 DashScope（通义千问）** 搭建的**支持多轮对话记忆的在线 RAG 问答服务**。
#简单概括：**读取私有知识库、结合历史对话，让大模型依据自有文档回答用户问题，缓解大模型幻觉；对话记录持久化保存，
# 同一个用户连续提问可以理解上下文。**
from langchain_core.documents import Document
# 操作系统路径工具，用于拼接会话存储目录
import os

# 导入自定义文件持久化聊天历史类，实现对话记录本地文件存储
from file_history_store import FileChatMessageHistory

# 会话历史存储路径：当前文件同级目录下 chat_histories 文件夹
# __file__ 当前脚本路径，dirname获取文件夹路径
HISTORY_STORE_PATH = os.path.join(os.path.dirname(__file__), "chat_histories")

def get_history(session_id: str) -> FileChatMessageHistory:
    """根据 session_id 返回对应的文件持久化聊天历史
    :param session_id: 用户会话唯一标识，区分不同用户对话
    :return: 文件存储对话历史实例
    """
    # 为每个会话创建独立对话存储器，会话数据保存在指定路径
    return FileChatMessageHistory(session_id=session_id, storage_path=HISTORY_STORE_PATH)

# 设置环境变量，DashScope API密钥，供通义千问模型、向量模型读取
os.environ["DASHSCOPE_API_KEY"] = "sk-59b7db233923469ba3e1a9b707125b2f"

# 字符串输出解析器：将LLM返回消息对象转为普通文本字符串
from langchain_core.output_parsers import StrOutputParser
# RunnablePassthrough：原样透传输入数据；RunnableLambda：封装自定义函数为链式组件
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory

# 导入向量存储服务（上一段代码封装的Chroma向量库）
from vector_stores import VectorStoreService
# DashScope向量嵌入模型
from langchain_community.embeddings import DashScopeEmbeddings
# 导入配置文件，读取模型名称、向量库参数等
import config_data as config
# 聊天提示词模板、消息占位符（用于填充对话历史）
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 通义千问对话大模型
from langchain_community.chat_models.tongyi import ChatTongyi

def print_prompt(prompt):
    """调试工具：打印组装完成后的完整Prompt，方便调试
    :param prompt: 填充变量后的提示词对象
    :return: 原prompt对象，保证链式调用不中断
    """
    print("="*20)
    # 转换为字符串打印完整prompt内容
    print(prompt.to_string())
    print("="*20)
    # 返回原对象，继续向后传递给LLM
    return prompt

# RAG核心服务类，封装检索、提示词、大模型、多轮对话记忆完整链路
class RagService(object):
    def __init__(self):
        # 初始化向量存储服务，传入嵌入模型（从配置读取向量模型名称）
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 构建对话提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                # 系统指令：优先使用检索到的参考资料回答
                ("system","以我提供的已知参考资料为主，"
                "简洁和专业的回答用户问题。参考资料:{context}。"),
                # 系统提示：告知模型存在历史对话
                ("system", "并且我提供用户的对话历史记录，如下："),
                # 消息占位符，自动填充多轮对话历史消息列表
                MessagesPlaceholder("history"),
                # 用户当前提问占位符
                ("user","请回答用户提问{input}")
            ]
        )

        # 初始化通义千问对话大模型，模型名称从配置读取
        self.chat_model = ChatTongyi(model=config.chat_model_name)

        # 组装完整RAG调用链，启动时自动构建
        self.chain = self.__get_chain()

    def __get_chain(self):
        """私有方法：组装完整带对话记忆的RAG链路"""
        # 从向量服务获取相似度检索器
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            """格式化检索返回文档列表，拼接成文本送入prompt
            :param docs: 向量库检索得到Document对象列表
            :return: 拼接完成的参考资料文本
            """
            # 检索无结果时返回默认提示
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            # 遍历每一条文档
            for doc in docs:
                # 拼接文档正文 + 文档元数据（来源、文件名等信息）
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value:dict)->str:#第一个转换的代码
            """提取用户query，传给检索器执行向量检索
            :param value: 上游传入字典
            :return: 用户原始提问文本
            """
            return value["input"]

        def format_for_prompt_template(value):#第二个转换的代码
            """数据结构适配：重整字典key，匹配prompt模板变量
            适配RunnableWithMessageHistory注入history后的数据结构
            """
            new_value = {}
            # 当前用户提问
            new_value["input"] = value["input"]["input"]
            # 检索得到的参考文档文本
            new_value["context"] = value["context"]
            # 自动注入的对话历史消息列表
            new_value["history"] = value["input"]["history"]
            return new_value

        # 构建基础RAG执行链
        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnableLambda(format_for_retriever) | retriever | format_document
                } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        # 包装为支持多轮对话记忆的链，自动读写会话历史
        conversation_chain = RunnableWithMessageHistory(
            # 基础RAG链
            chain,
            # 获取会话历史的回调函数
            get_history,
            # 输入消息对应的key（用户问题）
            input_messages_key="input",
            # 模板中对话历史占位符key
            history_messages_key="history",
        )

        return conversation_chain

    def retrieve(self, query: str, k: int = None) -> list[Document]:
        """
        对外提供的原始检索方法：返回相似度最高的文档列表（含元数据）
        供Web界面在问答命中时展示原图（读取元数据中的 image_path）
        :param query: 用户提问文本
        :param k: 检索返回条数，默认使用配置的 similarity_threshold
        :return: Document对象列表
        """
        # 未指定k时，使用配置文件中的默认检索条数
        search_k = k if k is not None else config.similarity_threshold
        # 用自定义k重新构造检索器并执行检索
        retriever = self.vector_service.vector_store.as_retriever(search_kwargs={"k": search_k})
        return retriever.invoke(query)

# 程序入口，直接运行文件时执行测试
if __name__ == '__main__':
    # 会话配置，指定当前会话ID，区分不同用户对话
    session_config = {
        "configurable":{
            "session_id":"user_001",
        }
    }

    # 实例化RAG服务，执行问答调用
    res = RagService().chain.invoke({"input":"我在学校参加华为的比赛有可能获奖吗"},session_config)
    # 打印模型最终回答
    print(res)
