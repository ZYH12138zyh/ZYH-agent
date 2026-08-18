# 知识库更新服务（离线版本）
import hashlib
import os
from datetime import datetime

# LangChain 向量库、嵌入模型、文本分割器
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入项目配置文件（向量库路径、分片参数、集合名称等）
import config_data as config

# 导入图片处理服务（照片入库时先存图 + OCR/描述提取文字）
from image_processor import ImageProcessor

# 初始化通义千问向量嵌入模型 text-embedding-v4
embedding_function = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key="sk-59b7db233923469ba3e1a9b707125b2f"  # 替换成你自己的API Key
)


def check_md5(md5_str: str) -> bool:
    """
    检查文本MD5是否已存在，用于去重
    :param md5_str: 待校验的MD5字符串
    :return: True=已存在；False=不存在
    """
    # 如果MD5记录文件不存在，先创建空文件，直接判定为不存在
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        # 逐行读取历史MD5列表进行比对
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False


def save_md5(md5_str: str):
    """
    将新文本MD5追加写入记录文件
    :param md5_str: 新增文档内容MD5
    """
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8') -> str:
    """
    计算字符串MD5哈希值，用于文档内容去重
    :param input_str: 原始文本内容
    :param encoding: 文本编码格式
    :return: MD5十六进制字符串
    """
    # 字符串转字节流
    # 将输入字符串按照指定编码转为字节数据，hashlib加密只接受bytes类型
    str_bytes = input_str.encode(encoding=encoding)
    # 创建一个md5加密对象
    md5_obj = hashlib.md5()
    # 向md5对象传入需要加密的字节内容
    md5_obj.update(str_bytes)
    # 获取加密后的十六进制字符串结果
    md5_hex = md5_obj.hexdigest()
    # 返回md5加密后的十六进制字符串
    return md5_hex


class KnowledgeBaseService(object):
    """
    知识库核心服务类
    封装：Chroma向量库初始化、文本分片、文档向量化入库、重复内容拦截
    """
    def __init__(self):
        # 创建向量持久化目录（不存在则自动创建）
        os.makedirs(config.persist_directory, exist_ok=True)

        # 初始化Chroma向量数据库实例
        self.chroma = Chroma(
            collection_name=config.collection_name,      # 向量集合名称（数据库的表名）
            embedding_function=embedding_function,        # 向量嵌入模型
            persist_directory=config.persist_directory,  # 向量持久化本地路径（数据库本地存储文件夹）
        )

        # 初始化递归文本分割器，用于长文本切分
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,                # 单个分片最大字符数（分割后的文本段最大长度）
            chunk_overlap=config.chunk_overlap,   # 分片之间重叠字符数（保证上下文连贯）（连续文本端之间的字符重叠数量）
            separators=config.separators,                # 优先分割符号列表（自然段落划分的符号）
            length_function=len,                         # 使用字符长度计算分片大小
        )

    def _embed_texts(self, data: str, filename: str, extra_metadata: dict = None) -> None:
        """
        公共入库方法：文本分片 → 组装元数据 → 向量化写入Chroma
        :param data: 待入库文本
        :param filename: 来源文件名（存入metadata溯源）
        :param extra_metadata: 额外元数据（如图片路径），会合并进默认元数据
        """
        # 判断文本长度，超长则进行分片；短文本直接作为单个片段
        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 元数据：保存来源文件、导入时间、操作人，后续检索可溯源
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "小曹",
        }
        # 合并额外元数据（例如图片入库时附带 image_path）
        if extra_metadata:
            metadata.update(extra_metadata)

        # 将分片文本批量写入向量库，每个分片绑定相同元数据
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

    def upload_by_str(self, data: str, filename: str) -> str:
        """
        通过文本字符串导入知识库主逻辑
        :param data: 原始文本字符串
        :param filename: 来源文件名（存入metadata溯源）
        :return: 执行结果提示信息
        """
        # 计算全文MD5，判断内容是否重复
        md5_hex = get_string_md5(data)

        # MD5命中，说明文档内容已入库，直接跳过
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        # 调用公共入库方法
        self._embed_texts(data, filename)

        # 入库成功后保存MD5，避免下次重复导入
        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"

    def upload_by_image(self, image_bytes: bytes, filename: str) -> str:
        """
        通过图片导入知识库主逻辑：存图 + 提取文字 + 向量化入库
        :param image_bytes: 图片二进制内容
        :param filename: 原始图片文件名（存入metadata溯源）
        :return: 执行结果提示信息
        """
        # 计算图片字节的MD5，用于判断该照片是否已入库（避免重复识别）
        md5_hex = hashlib.md5(image_bytes).hexdigest()

        # MD5命中，说明该照片已入库，直接跳过
        if check_md5(md5_hex):
            return "[跳过]这张照片已经存在知识库中"

        # 调用图片处理服务：存图 + OCR提取文字（文字太少则视觉描述兜底）
        text, image_path = ImageProcessor().process_image(image_bytes, filename)

        # 识别失败（文字为空）时不入库，直接返回失败提示
        if not text:
            return "[失败]图片未能提取到有效内容"

        # 调用公共入库方法，额外附带图片本地路径，供问答命中时展示原图
        self._embed_texts(text, filename, extra_metadata={"image_path": image_path})

        # 入库成功后保存MD5，避免下次重复导入
        save_md5(md5_hex)
        return "[成功]照片内容已经成功载入向量库"


if __name__ == '__main__':
    # 程序入口：本地批量读取txt文件，批量导入知识库
    service = KnowledgeBaseService()
    # 待导入文件列表
    file_list = [
        "01_招生政策.txt",
        "02_院校专业介绍.txt",
        "03_历年录取分数线.txt",
        "05_校园规章制度.txt",
        "竞赛获奖.txt"
    ]
    # 循环逐个读取文件并上传
    for filename in file_list:
        with open(f"data/{filename}", "r", encoding="utf-8") as f:
            content = f.read()
        r = service.upload_by_str(data=content, filename=filename)
        print(f"{filename} 上传结果：{r}")
