# Python For Data Science: Lab02
# Ho ten: Ton Thien Minh Anh
# MSSV: 18110049
# Bai 3
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os 

df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week2/creditcard.csv')

# print(df.info())
# print(df.describe())

# Correlation Map
plt.figure(figsize=(32,32))
sns.heatmap(df.corr(),linewidth=.1,cmap="YlGnBu",annot=True)
plt.yticks(rotation=0)

# Dua vao Correlation Heatmap, ta chon 2 features la Class va V11 vi chung co moi tuong quan voi nhau tuong doi cao

# Box Plot
plt.figure(figsize=(10,10))
Raise_hand=sns.boxplot(x="Class",y="V11",data=df)
plt.show()
# Facetgrid
Facetgrid=sns.FacetGrid(df,hue="Class",size=6)
Facetgrid.map(sns.kdeplot,"V11",shade=True)
Facetgrid.set(xlim=(0,df['V11'].max()))
Facetgrid.add_legend()

# Nhan xet Box plot va Facetgrid:
# 1/ Dua vao do thi, ta thay duoc rang co su tuong quan nhat dinh giua Class va V11
# 2/ Doi voi cac truong hop Class = 0 (fraud), chi so V11 thuong roi vao khoang tu 0 den 2 va do thi co xu huong giam dan khi V11 cang tang.
# 3/ Doi voi cac truong hop Class = 1 (otherwise), do thi co xu huong tang dan tu khoang 0 den 3 va cham dinh o khoang 4, sau do thi giam dan. 
# Ket luan: Tu cac yeu to tren, co the tam thoi xac dinh rang ti trong chi so V11 o muc thap thi kha nang xay ra gian lan the tin dung (class = 0, fraud) cang cao. 

# Scatter Plot
sns.set_style("whitegrid")
sns.FacetGrid(df, hue = "Class" , height = 6).map(plt.scatter,"Class","V11").add_legend()
plt.show()

# Nhan xet: 
# 1/ Doi voi bieu do Scatter Plot, ta co the thay duoc cac phan gia tri am.
# 2/ Class = 0 thuong co chi so V11 chiem da so trong khoang gia tri tu -5 den 5.
# 3/ Class = 1 thuong co it chi so V11 la gia tri am hon, dong thoi gia tri V11 day dac o khoang tu 0 den 10.