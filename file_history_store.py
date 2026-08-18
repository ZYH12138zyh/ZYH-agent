# 长期会话记忆存储服务（在线版本）
from __future__ import annotations
import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    基于本地JSON文件实现的LangChain会话记忆持久化类
    继承BaseChatMessageHistory，实现会话消息本地文件读写存储
    """
    def __init__(self, session_id, storage_path):
        """
        初始化会话存储实例
        :param session_id: 会话唯一ID，作为json文件名
        :param storage_path: 消息存储的根文件夹路径
        """
        self.session_id = session_id      # 会话唯一标识
        self.storage_path = storage_path  # 存储根目录路径
        # 拼接完整文件路径：存储目录/会话ID.json
        self.file_path = os.path.join(self.storage_path, self.session_id + ".json")

        # 自动创建文件夹，若目录不存在则生成，已存在不会报错
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        追加多条消息到会话历史文件
        :param messages: BaseMessage消息对象序列（HumanMessage/AIMessage等）
        """
        # 读取当前已保存全部历史消息
        all_messages = list(self.messages)
        # 将新消息追加至消息列表末尾
        all_messages.extend(messages)

        # LangChain消息对象序列化：转为字典结构，方便JSON保存
        new_messages = [message_to_dict(message) for message in all_messages]
        # 写入JSON文件，utf-8编码，关闭ascii转义，自动缩进方便阅读
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False, indent=2)

    @property
    def messages(self) -> list[BaseMessage]:
        """
        属性装饰器：读取当前会话所有历史消息
        @return: BaseMessage 对象列表
        """
        try:
            # 读取本地json文件
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
            # 将字典格式消息反序列化为LangChain消息对象
            return messages_from_dict(messages_data)
        except FileNotFoundError:
            # 文件不存在时，代表暂无历史会话，返回空列表
            return []

    def clear(self) -> None:
        """清空当前会话全部历史消息"""
        # 写入空列表，覆盖原有内容，实现清空会话
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
