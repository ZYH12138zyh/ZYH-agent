#向量存储服务（在线版本）
# 导入Chroma向量数据库实现类
from langchain_chroma import Chroma
# 导入配置模块，存放集合名称、持久化目录、检索参数等配置常量
import config_data as config

# 向量存储服务类，封装Chroma向量库初始化、检索器获取逻辑
class VectorStoreService(object):
    # 构造方法，接收嵌入模型实例
    def __init__(self, embedding):
        # 将传入的嵌入模型保存为实例属性
        self.embedding = embedding

        # 初始化Chroma向量数据库实例
        self.vector_store = Chroma(
            # 向量集合名称，从配置文件读取
            collection_name=config.collection_name,
            # 指定向量计算使用的嵌入函数
            embedding_function=self.embedding,
            # 向量持久化本地文件夹路径，数据落盘保存
            persist_directory=config.persist_directory,
        )

    # 获取检索器对象方法，供后续RAG检索调用
    def get_retriever(self):
        # 将向量库转为检索器，search_kwargs设置检索参数
        # k：一次检索返回top-k相似度最高的文档
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})

# 程序入口，仅直接运行该文件时执行下面代码
if __name__ == '__main__':
    # 导入通义千问DashScope嵌入模型
    from langchain_community.embeddings import DashScopeEmbeddings

    # 实例化阿里云通义嵌入模型
    embedding = DashScopeEmbeddings(
        # 使用text-embedding-v4向量模型
        model="text-embedding-v4",
        # DashScope接口密钥，用于调用向量服务
        dashscope_api_key="sk-59b7db233923469ba3e1a9b707125b2f"
    )
    # 实例化向量存储服务，并获取检索器
    retriever = VectorStoreService(embedding).get_retriever()

    # 执行相似度检索，传入用户查询文本
    res = retriever.invoke("我在学校参加华为的比赛有可能获奖吗")
    # 打印检索返回的文档结果
    print(res)
