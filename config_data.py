#配置文件
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

md5_path = os.path.join(_BASE_DIR, "md5.text")

collection_name="rag"
persist_directory = os.path.join(_BASE_DIR, "chroma_db")

chunk_size=1000
chunk_overlap=100
separators=["\n\n","\n",".","!","?","。","!","?"," ",""]
max_split_char_number=1000

similarity_threshold=2

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}

# ===== 图片入库相关配置 =====
# DashScope API密钥，供通义千问视觉模型读取（与项目其余模块共用同一个Key）
dashscope_api_key = "sk-59b7db233923469ba3e1a9b707125b2f"
# 图片本地保存目录（原图落盘位置）
image_dir = os.path.join(_BASE_DIR, "data", "images")
# 图片OCR识别使用的视觉模型
vl_ocr_model = "qwen-vl-ocr-latest"
# 图片视觉描述兜底使用的视觉模型
vl_desc_model = "qwen-vl-plus"
# OCR识别文字长度小于该阈值时，改用视觉模型描述图片内容
image_text_min_length = 10
