# 导入streamlit库
import streamlit as st

# 导入其他必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置页面标题
st.title('Streamlit 示例')

# 在页面上写文本
st.write("这是一个简单的Streamlit应用，用于展示基本功能。")

# 文本输入
user_input = st.text_input("请输入您的名字", '')

# 显示输入的文本
st.write(f'你好, {user_input}!')

# 数字输入
number = st.number_input('输入一个数字')

# 显示数字的平方
st.write(f'数字的平方是: {number * number}')

# 数据生成
data = pd.DataFrame({
  '第一列': range(1, 101),
  '第二列': np.random.randn(100).cumsum()
})

# 数据表展示
st.write("这是一个数据表:", data)

# 绘制折线图
st.line_chart(data)

# 绘制直方图
st.bar_chart(data['第二列'])

# 使用matplotlib绘图
fig, ax = plt.subplots()
ax.hist(data['第二列'], bins=20)

st.pyplot(fig)
