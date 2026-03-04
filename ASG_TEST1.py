# %% [markdown]
# # 儿童故事生成应用
# 学号：XXXXXXXXX
# 
# 本应用基于Hugging Face Transformers，实现：
# 1. 图片上传与描述生成 (BLIP模型)
# 2. 故事生成 (GPT-2)
# 3. 语音合成 (Bark TTS)

# %% [markdown]
# ## 1. 安装依赖

# %%
# 安装必要的包
!pip install -q streamlit transformers torch accelerate pillow bark

# %% [markdown]
# ## 2. 导入库

# %%
import streamlit as st
from PIL import Image
from transformers import pipeline
import tempfile
import os
import numpy as np

# %% [markdown]
# ## 3. 定义核心功能类

# %%
class StorytellingApp:
    """故事生成应用的核心功能类"""
    
    def __init__(self):
        self.captioner = None
        self.story_generator = None
        self.tts = None
    
    def load_models(self):
        """加载所有模型（带缓存）"""
        if self.captioner is None:
            with st.spinner("正在加载图像描述模型..."):
                self.captioner = pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-base"
                )
        
        if self.story_generator is None:
            with st.spinner("正在加载故事生成模型..."):
                self.story_generator = pipeline(
                    "text-generation",
                    model="gpt2",
                    max_new_tokens=150,
                    temperature=0.8
                )
        
        if self.tts is None:
            with st.spinner("正在加载语音合成模型..."):
                self.tts = pipeline(
                    "text-to-speech",
                    model="suno/bark"
                )
        
        st.success("所有模型加载完成！")
        return self
    
    def generate_caption(self, image):
        """生成图片描述"""
        result = self.captioner(image)
        return result[0]['generated_text']
    
    def generate_story(self, caption):
        """基于描述生成故事"""
        prompt = f"根据这个场景给小朋友讲一个小故事：{caption}。故事要简单有趣，适合3-10岁孩子。"
        result = self.story_generator(prompt)
        story = result[0]['generated_text']
        # 移除提示词部分
        story = story.replace(prompt, "").strip()
        return story
    
    def text_to_speech(self, text):
        """将文本转为语音"""
        audio = self.tts(text)
        return audio['audio']

# %% [markdown]
# ## 4. Streamlit 界面代码

# %%
def main():
    """主应用函数"""
    st.set_page_config(
        page_title="儿童故事生成器",
        page_icon="📖",
        layout="wide"
    )
    
    st.title("📖 儿童故事生成器")
    st.markdown("上传一张图片，AI会为你生成一个小故事并朗读出来！")
    
    # 初始化应用
    app = StorytellingApp()
    
    # 侧边栏
    with st.sidebar:
        st.header("关于本应用")
        st.write("""
        这是一个为3-10岁儿童设计的互动故事生成器。
        
        **使用步骤：**
        1. 上传一张图片
        2. AI分析图片内容
        3. 生成趣味故事
        4. 收听语音朗读
        
        **技术栈：**
        - Hugging Face Transformers
        - BLIP图像描述模型
        - GPT-2故事生成
        - Bark语音合成
        """)
        
        # 模型加载按钮
        if st.button("🚀 加载AI模型"):
            with st.spinner("正在加载模型（首次运行需要下载）..."):
                app.load_models()
                st.session_state['app'] = app
                st.success("模型已加载！")
    
    # 主界面
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 上传图片")
        uploaded_file = st.file_uploader(
            "选择一张图片",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="图片应放在当前目录或直接上传"
        )
        
        if uploaded_file is not None:
            # 显示图片
            image = Image.open(uploaded_file)
            st.image(image, caption='你上传的图片', use_column_width=True)
            st.session_state['image'] = image
    
    with col2:
        st.subheader("2. 生成的故事")
        
        # 检查模型是否已加载
        if 'app' not in st.session_state:
            st.info("请先在侧边栏加载AI模型")
        elif 'image' not in st.session_state:
            st.info("请先上传图片")
        else:
            # 生成按钮
            if st.button("✨ 生成故事"):
                app = st.session_state['app']
                image = st.session_state['image']
                
                try:
                    # 步骤1：生成描述
                    with st.spinner("正在分析图片..."):
                        caption = app.generate_caption(image)
                    st.success("✅ 图片分析完成")
                    st.write(f"**图片描述**：{caption}")
                    
                    # 步骤2：生成故事
                    with st.spinner("正在创作故事..."):
                        story = app.generate_story(caption)
                    st.success("✅ 故事创作完成")
                    st.write(f"**故事内容**：{story}")
                    
                    # 步骤3：生成语音
                    with st.spinner("正在合成语音..."):
                        audio = app.text_to_speech(story)
                    
                    # 保存临时音频文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                        if isinstance(audio, np.ndarray):
                            # 如果返回的是numpy数组，需要转换
                            import scipy.io.wavfile as wav
                            wav.write(f.name, 24000, audio)
                        else:
                            f.write(audio)
                        
                        # 显示音频播放器
                        st.audio(f.name, format='audio/wav')
                    
                    st.success("✅ 语音合成完成！")
                    
                except Exception as e:
                    st.error(f"出错了: {str(e)}")

# %% [markdown]
# ## 5. 运行应用

# %%
# 检查是否在Streamlit环境中运行
import sys

if __name__ == "__main__":
    # 如果在Jupyter中运行，显示说明
    if 'ipykernel' in sys.modules:
        st.markdown("""
        ### 🚀 如何运行这个应用
        
        **方法1：在本地运行**
        ```bash
        # 保存本notebook为Python脚本
        jupyter nbconvert --to script <你的学号>_asg.ipynb
        
        # 运行Streamlit
        streamlit run ASG_TEST1.py
