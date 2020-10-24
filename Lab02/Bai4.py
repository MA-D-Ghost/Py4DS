# Python For Data Science: Lab02
# Ho ten: Ton Thien Minh Anh
# MSSV: 18110049
# Bai 4
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os 

df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week2/HappinessReport2020.csv')


print(df.info())
# Sau khi dung lenh info, ta xac dinh duoc data khong co gia tri null

# Loai bo cac cot khong su dung 
df = df.drop(['Standard error of ladder score', 'upperwhisker',
 'upperwhisker', 'lowerwhisker', 'Explained by: Log GDP per capita',
  'Explained by: Social support', 'Explained by: Healthy life expectancy',
   'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption'], axis=1)

print(df.head())
print(df.shape)

# Kiem tra do thi cua tung gia tri
plt.rcParams['figure.figsize'] = (12, 12)
df.hist()

# Ve Correlation Heatmap
plt.figure(figsize=(11,11))
sns.heatmap(df.corr(),linewidth=.1,cmap="YlGnBu",annot=True)
plt.yticks(rotation=0)

# Dua vao Heatmap, ta thay nhung features nhu Ladder score, Logged GDP per capita, Social support, Healthy life expectancy deu co moi tuong quan mat thiet voi nhau.
# Tu day, ta co duoc thong tin rang muon cho nguoi dan quoc gia cua minh hanh phuc, can phai tap trung vao viec phat trien va dam bao duoc viec lam, hoat dong xa hoi, he thong y te,... cua dat nuoc. 



# KDE Plot
Facetgrid=sns.FacetGrid(df,hue="Regional indicator",size=6)
Facetgrid.map(sns.kdeplot,"Ladder score",shade=True)
Facetgrid.set(xlim=(0,df['Ladder score'].max()))
Facetgrid.add_legend()

# Scatter Plot
sns.set_style("whitegrid")
sns.FacetGrid(df, hue = "Regional indicator" , height = 6).map(plt.scatter,"Regional indicator","Ladder score").add_legend()
plt.show()

# Nhan xet: 
# 1/ Western Europe la khu vuc dan dau, co diem Ladder score trung binh o muc cao, phan bo o muc 7 den 8.
# 2/ Sub-Saharan Africa la khu vuc dong quoc gia nhung diem chi phan bo o khoang tu 4 den 5 => Khu vuc nay da phan la cac nuoc con dang phat trien, he thong y te, xa hoi con chua phat trien va GDP thap. 
# 3/ South Asia la khu vuc co quoc gia diem thap nhat nam o muc duoi 3.  

