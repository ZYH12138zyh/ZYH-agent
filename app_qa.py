#项目主程序，启动对话WEB界面
import os
import sys
# 关闭Chroma向量数据库遥测上报（关闭匿名数据收集）
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# 伪造posthog空模块，规避langchain内部自动导入posthog引发的报错
sys.modules["posthog"] = type("FakePosthog", (), {})()

import time
# 导入自定义RAG检索增强服务类
from rag import RagService
# 导入图片处理服务（聊天输入图片时，先把图片转成文字）
from image_processor import ImageProcessor
# 导入Streamlit网页框架
import streamlit as st

# 设置网页标题
st.title("家里蹲大学官网")
# 绘制分割线
st.divider()

# 实例化RAG服务（临时实例，后续会使用session_state持久化版本）
rag_service=RagService()

# 初始化会话状态：如果不存在message键，创建聊天历史列表
if "message" not in st.session_state:
    # 初始消息：AI开场白
    st.session_state["message"]=[{"role":"assistant","content":"你好，有什么可以帮助你？"}]

# 在session_state中缓存RAG实例，避免每次交互重复初始化
if "rag" not in st.session_state:
    st.session_state["rag"]=RagService()

# 循环渲染历史聊天记录
for message in st.session_state["message"]:
    # 根据角色渲染对应气泡
    st.chat_message(message["role"]).write(message["content"])

# 创建聊天输入框：支持输入文字，也支持附带图片附件
prompt = st.chat_input(
    "请输入问题，也可附上图片",
    accept_file=True,                    # 开启附件上传能力
    file_type=["jpg", "jpeg", "png"],    # 限制附件仅支持图片格式
)

# 用户提交内容不为空时执行
if prompt:
    # accept_file=True 时，prompt 返回一个字典类对象，含 text（文字）和 files（图片列表）
    text = prompt.text or ""       # 用户输入的文字（纯图片时为空字符串）
    files = prompt.files or []     # 用户上传的图片文件列表

    # 图片内容识别：把图片转成文字（OCR，文字太少则视觉描述兜底）
    image_content = ""
    image_file = None
    if files:
        # 只处理第一张图片
        image_file = files[0]
        with st.spinner("识别图片中..."):
            try:
                # 复用图片处理服务，返回(识别文字, 图片路径)；这里只用识别文字
                image_content, _ = ImageProcessor().process_image(image_file.getvalue(), image_file.name)
            except Exception:
                image_content = ""  # 识别失败不影响主流程，退化为纯文字提问

    # 拼接最终查询文本：图片内容 + 用户文字
    if image_content and text:
        query = f"图片内容：{image_content}\n用户问题：{text}"
    elif image_content:
        query = image_content
    else:
        query = text

    # 兜底：图片识别失败且无文字时，提示用户，不做无效查询
    if not query.strip():
        st.chat_message("assistant").write("抱歉，图片没识别出内容，请补充文字描述或换张清晰的照片。")
        st.stop()

    # 渲染用户消息气泡（文字 + 图片）
    user_bubble = st.chat_message("user")
    if text:
        user_bubble.write(text)
    if image_file is not None:
        # width 限制展示宽度（像素），避免上传原图在聊天气泡里撑满整页
        user_bubble.image(image_file, width=250)
    # 将用户消息存入会话历史（历史记录只存文字，图片不跨刷新持久化）
    st.session_state["message"].append({"role": "user", "content": text or "[图片]"})

    # 检索命中的照片：从向量库检索到的文档中提取 image_path，命中则展示原图
    image_paths = []
    try:
        # 用稍大的k检索，更稳定命中含图片的片段
        docs = st.session_state["rag"].retrieve(query, k=4)
        for doc in docs:
            # 从元数据读取图片路径（纯文本片段没有 image_path 字段）
            img_path = doc.metadata.get("image_path")
            # 图片路径存在、文件确实存在且未重复展示时收集
            if img_path and os.path.exists(img_path) and img_path not in image_paths:
                image_paths.append(img_path)
    except Exception:
        pass  # 检索失败不阻塞主问答流程

    # 在回答前展示命中的原图
    for img_path in image_paths:
        st.image(img_path, caption=f"命中图片：{os.path.basename(img_path)}")

    # 列表缓存AI流式返回的分片文本
    ai_res_list=[]
    # 显示加载等待提示
    with st.spinner("AI思考中..."):
        # LangChain会话配置，指定session_id，用于绑定记忆持久化
        session_config = {"configurable": {"session_id": "chat_session_01"}}
        # 调用chain流式接口，获取生成器流数据
        res_stream = st.session_state["rag"].chain.stream({"input": query}, config=session_config)

        # 包装生成器：一边流式输出，一边把分片存入缓存列表
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # 流式渲染AI回答到页面
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        # 将所有分片拼接成完整字符串，存入会话历史，实现刷新页面不丢失对话
        st.session_state["message"].append({"role":"assistant","content":"".join(ai_res_list)})
