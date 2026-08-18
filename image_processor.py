# 图片处理服务（图片入库的预处理模块）
# 作用：把用户上传的照片「存图 + 提取文字」——先OCR识别图中文字，文字太少时用视觉模型补充描述
# 最终把照片转成一段文本，交给知识库的向量化入库流程处理
import os
from datetime import datetime

# 阿里云 DashScope 多模态会话接口，用于调用通义千问视觉模型（OCR / 图片描述）
from dashscope import MultiModalConversation

# 导入项目配置文件（图片保存目录、视觉模型名、OCR兜底阈值等）
import config_data as config


class ImageProcessor(object):
    """
    图片处理服务类
    封装：图片落盘、OCR文字识别、视觉描述兜底
    """

    @staticmethod
    def save_image(image_bytes: bytes, filename: str) -> str:
        """
        将图片字节保存到本地目录，返回绝对路径
        :param image_bytes: 图片二进制内容
        :param filename: 原始文件名（用于提取扩展名）
        :return: 保存后的图片绝对路径
        """
        # 确保图片目录存在（不存在则自动创建）
        os.makedirs(config.image_dir, exist_ok=True)

        # 提取原始文件扩展名（如 .jpg / .png），统一转小写
        ext = os.path.splitext(filename)[1].lower() or ".jpg"

        # 用时间戳生成唯一文件名，避免重名覆盖（微秒级精度保证不重复）
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext
        abs_path = os.path.join(config.image_dir, unique_name)

        # 以二进制写模式落盘
        with open(abs_path, "wb") as f:
            f.write(image_bytes)

        return abs_path

    @staticmethod
    def _call_vision(model: str, image_path: str, prompt: str, **kwargs) -> str:
        """
        调用通义千问视觉模型的通用方法
        :param model: 视觉模型名称（OCR模型或描述模型）
        :param image_path: 本地图片绝对路径
        :param prompt: 发给模型的文字指令
        :param kwargs: 透传给 DashScope 接口的额外参数（如 ocr_options）
        :return: 模型返回的文本内容
        """
        # 组装多模态消息：图片用本地文件路径（file:// 协议），SDK会自动上传
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{image_path}"},
                    {"text": prompt},
                ],
            }
        ]

        # 调用 DashScope 多模态接口（api_key 从配置读取）
        response = MultiModalConversation.call(
            model=model,
            api_key=config.dashscope_api_key,
            messages=messages,
            **kwargs,
        )

        # 解析返回结构：output.choices[0].message.content[0]["text"]
        # content 是列表，其中 text 字段所在项才是模型输出的文字
        try:
            content = response.output.choices[0].message.content
            # 遍历 content 列表，拼接所有 text 字段（正常情况下只有一项）
            text_parts = [item["text"] for item in content if "text" in item]
            return "".join(text_parts).strip()
        except (AttributeError, KeyError, IndexError, TypeError):
            # 解析失败时返回空字符串，交由上层兜底处理
            return ""

    @staticmethod
    def ocr_image(image_path: str) -> str:
        """
        识别图片中的文字（OCR）
        :param image_path: 图片绝对路径
        :return: 识别出的文字内容
        """
        return ImageProcessor._call_vision(
            model=config.vl_ocr_model,
            image_path=image_path,
            prompt="请提取图片中的全部文字，不要添加额外描述和格式",
            ocr_options={"task": "text_recognition"},
        )

    @staticmethod
    def describe_image(image_path: str) -> str:
        """
        描述图片内容（OCR识别不到文字时的兜底方案）
        :param image_path: 图片绝对路径
        :return: 图片内容描述文本
        """
        return ImageProcessor._call_vision(
            model=config.vl_desc_model,
            image_path=image_path,
            prompt="请用一段话详细描述这张图片的内容（场景、主体、文字等）",
        )

    def process_image(self, image_bytes: bytes, filename: str) -> tuple:
        """
        图片处理主逻辑：存图 → OCR提取文字 → 文字太少则视觉描述兜底
        :param image_bytes: 图片二进制内容
        :param filename: 原始文件名
        :return: (最终文本内容, 图片本地绝对路径)
        """
        # 第一步：把图片保存到本地，拿到路径
        image_path = self.save_image(image_bytes, filename)

        # 第二步：OCR识别图片中的文字
        text = self.ocr_image(image_path)

        # 第三步：识别文字太少（如纯风景照、人物照），改用视觉模型描述图片
        if len(text) < config.image_text_min_length:
            text = self.describe_image(image_path)

        return text, image_path


if __name__ == '__main__':
    # 程序入口：本地测试单张图片的处理流程
    processor = ImageProcessor()
    # 替换为本地测试图片路径
    test_image = os.path.join(config.image_dir, "test.jpg")
    if os.path.exists(test_image):
        with open(test_image, "rb") as f:
            data = f.read()
        result_text, result_path = processor.process_image(data, "test.jpg")
        print(f"图片保存路径：{result_path}")
        print(f"提取/描述结果：{result_text}")
    else:
        print(f"测试图片不存在：{test_image}")
