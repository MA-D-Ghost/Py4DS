# Python For Data Science: Lab02
# Ho ten: Ton Thien Minh Anh
# MSSV: 18110049
# Bai 1
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os 

df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week2/xAPI-Edu-Data.csv')
print(df.head())

for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)

sns.pairplot(df,hue="Class")

# Plot label 
# Kiem tra phan bo cua labels co deu hay khong?
# Neu can bang => Co the su dung truc tiep duoc
P_satis=sns.countplot(x="Class",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))


df.Class.value_counts(normalize=True).plot(kind="bar")

# Heatmap 
# The hien moi tuong quan giua cac features
# Neu features co moi tuong quan lon => Kiem tra moi tuong quan cua rieng 2 features nay de loc data

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidth=.1,cmap="YlGnBu",annot=True)
plt.yticks(rotation=0)


print(df.Class.value_counts())
print(df.Class.value_counts(normalize=True))

plt.subplots(figsize=(20,8))
df["raisedhands"].value_counts().sort_index().plot.bar()
plt.title("No. of times",fontsize=18)
plt.xlabel("No. of times, student raised their hand",fontsize=14)
plt.ylabel("No. of student, on particular times",fontsize=14)
plt.show()

# Box Plot
plt.figure(figsize=(10,10))
Raise_hand=sns.boxplot(x="Class",y="raisedhands",data=df)
plt.show()
# Nhan xet bieu do: Cac hoc sinh trong lop High Level (H) co so lan gio tay nhieu hon 2 lop Middle-Level va Low-Level


Facetgrid=sns.FacetGrid(df,hue="Class",size=6)
Facetgrid.map(sns.kdeplot,"raisedhands",shade=True)
Facetgrid.set(xlim=(0,df['raisedhands'].max()))
Facetgrid.add_legend()

# Data.groupby
print(df.groupby(["ParentschoolSatisfaction"])["Class"].value_counts())
print(pd.crosstab(df['Class'],df['ParentschoolSatisfaction']))

sns.countplot(x="ParentschoolSatisfaction",data=df,hue="Class")

# Nhan xet: Phu huynh trong lop High-Level thuong hai long ve truong hoc hon

# Pie chart
labels=df.ParentschoolSatisfaction.value_counts()
colors=["blue","green"]
explode = [0,0]
sizes = df.ParentschoolSatisfaction.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels =labels,colors=colors,autopct='%1.1f%%')
plt.title("Parent school Satisfaction in Data",fontsize=15)
plt.show()
