# 基础镜像：Python 3.10 轻量版
FROM python:3.10-slim

# 设置容器内工作目录
WORKDIR /app

# 先复制依赖文件，利用Docker缓存加速
COPY requirements.txt .

# 安装依赖，国内源加速
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 暴露 Gradio 默认端口
EXPOSE 7860

# 容器启动命令：对应你的网页脚本文件名
CMD ["python", "rag_webui.py"]