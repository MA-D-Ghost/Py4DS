# Python For Data Science: Lab02
# Ho ten: Ton Thien Minh Anh
# MSSV: 18110049
# Bai 2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os 

df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week2/xAPI-Edu-Data.csv')
df_label = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week2/xAPI-Edu-Data.csv')

#Label Encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for column in df.columns: 
   df_label[column]=labelencoder.fit_transform(df[column])


# Ve Correlation Heatmap
sns.heatmap(df_label.corr(),cmap="YlGnBu")

# Dua vao moi tuong quan giua 2 features trong Correlation Heatmap, ta chon 2 features la Class (object) va Discussion (int64)

# Box Plot
plt.figure(figsize=(10,10))
Raise_hand=sns.boxplot(x="Class",y="Discussion",data=df)
plt.show()
# Nhan xet bieu do: Dua vao bieu do Box Plot, ta co the thay duoc hoc sinh trong lop High-Level thuong tham gia vao cac nhom thao luan ve mon hoc nhieu hon  


# Facetgrid
Facetgrid=sns.FacetGrid(df,hue="Class",size=6)
Facetgrid.map(sns.kdeplot,"Discussion",shade=True)
Facetgrid.set(xlim=(0,df['Discussion'].max()))
Facetgrid.add_legend()

# Scatter Plot

sns.set_style("whitegrid")
sns.FacetGrid(df, hue = "Class" , height = 6).map(plt.scatter,"Class","Discussion").add_legend()
plt.show()

# Nhan xet bieu do Facetgrid va Scatter Plot:
# 1/ Hoc sinh trong lop Low-Level thuong it tham gia vao cac nhom thao luan ve mon hoc, da phan so lan tham gia cua du lieu chu yeu roi vao khoang 0 den 40.
# 2/ Hoc sinh lop Middle-Level tham gia tuong doi vao cac nhom thao luan ve mon hoc, chiem da so o muc 20 den 40 lan, sau do do thi giam dan. 
# 3/ Hoc sinh lop High-Level chiem nhieu nhat o so lan tham gia vao nhom thao luan muc 60 den 80.
# Ket luan: Nho tham gia vao cac nhom thao luan ve bai hoc, hoc sinh co the dat diem so cao hon, dong nghia voi viec ti le duoc vao hoc o cac lop High-Level se tang. 




# Data.groupby & countplot
print(df.groupby(["StudentAbsenceDays"])["Class"].value_counts())
print(pd.crosstab(df['Class'],df['StudentAbsenceDays']))

sns.countplot(x="StudentAbsenceDays",data=df,hue="Class")
# Nhan xet:
# 1/ Hoc sinh lop Low-Level thuong co so lan nghi hoc nhieu hon 7 ngay.
# 2/ Hoc sinh cua lop Middle-Level va High-Level thuong co so lan nghi hoc it hon 7 ngay. 
# Ket luan: Nhu vay, ta co the tam thoi xac dinh rang, so lan nghi hoc ti le nghich voi loai lop hoc. Cang nghi hoc it thi se khong bi mat kien thuc => Diem cao hon => Co hoi duoc hoc o lop Middle va High-Level cao hon.  

# Pie chart
labels=df.StudentAbsenceDays.value_counts()
colors=["orange","green"]
explode = [0,0]
sizes = df.StudentAbsenceDays.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels =labels,colors=colors,autopct='%1.1f%%')
plt.title("Student Absence Days in Data",fontsize=15)
plt.show()