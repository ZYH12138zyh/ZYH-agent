# 知识库的更新主程序（离线版本）
import streamlit as st
import time
# 导入自定义知识库业务类，封装文档切片、向量化、入库等逻辑
from knowledge_base import KnowledgeBaseService

# 页面标题
st.title("知识库更新服务")

# 文件上传组件

uploader_file = st.file_uploader(
    label="请上传TXT文件",
    type=['txt'],                # 限制文件后缀仅txt
    accept_multiple_files=False, # 关闭多文件上传，只允许单次上传一个文件
)

# 图片上传组件（照片入库：OCR/视觉描述提取文字后进入知识库）
uploader_image = st.file_uploader(
    label="请上传照片（PNG/JPG）",
    type=['png', 'jpg', 'jpeg'], # 限制可上传的图片后缀
    accept_multiple_files=False, # 关闭多文件上传，只允许单次上传一张
)

# 初始化知识库服务实例（存入session_state，保证页面刷新不重复创建实例）
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

# 判断：检测到用户上传文件后执行后续逻辑
if uploader_file is not None:
    # 获取上传文件名称
    file_name = uploader_file.name
    # 获取文件MIME类型
    file_type = uploader_file.type
    # 获取文件字节大小，转换单位KB
    file_size = uploader_file.size / 1024

    # 展示文件基础信息
    st.subheader(f"文件名:{file_name}")
    st.write(f"格式:{file_type}|大小:{file_size:.2f} KB")

    # 将上传文件二进制内容解码为utf-8文本字符串
    text = uploader_file.getvalue().decode("utf-8")

    # 加载动画：执行知识库导入逻辑
    with st.spinner("载入知识库中。。。"):
        time.sleep(1)  # 模拟加载延时，可删除
        # 调用知识库服务方法，传入文本内容与文件名，执行文档入库
        result = st.session_state["service"].upload_by_str(text, file_name)
        #.upload_by_str这个用到的是knowledge_base.py中的upload_by_str方法
        # 输出执行结果
        st.write(result)

# 判断：检测到用户上传图片后执行后续逻辑
if uploader_image is not None:
    # 获取上传图片名称
    image_name = uploader_image.name

    # 展示图片基础信息
    st.subheader(f"图片名:{image_name}")
    st.write(f"格式:{uploader_image.type}|大小:{uploader_image.size / 1024:.2f} KB")

    # 加载动画：执行图片入库逻辑（含OCR/视觉描述，耗时较长）
    with st.spinner("照片识别入库中。。。"):
        # 调用知识库服务方法，传入图片二进制与文件名，执行照片入库
        result = st.session_state["service"].upload_by_image(uploader_image.getvalue(), image_name)
        # 输出执行结果
        st.write(result)
