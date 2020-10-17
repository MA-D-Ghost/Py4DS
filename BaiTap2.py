# Python for Data Science: Week 1
# MSSV: 18110049
# Ho ten: Ton Thien Minh Anh 

# Load libraries
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier #Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation
import eli5 #Calculating and Displaying importance using the eli5 library
from eli5.sklearn import PermutationImportance

col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
# load dataset
df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week1/diabetes.csv',header=0,names=col_names)

print(df.head())
print(df.info())

# 2. FEATURE SELECTION
#split dataset in features and target variable 
feature_cols = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']
X = df[feature_cols] #Features
y = df.label # Target variable

# 3. SPLITTING DATA
# Split dataset into training set and test set
# 70% training and 30% test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# 4. BUILDING DECISION TREE MODEL
# Create Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf=clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

perm = PermutationImportance(clf,random_state=1).fit(X_test,y_test)
eli5.show_weights(perm,feature_names=X_test.columns.tolist())

# 5.EVALUATING MODEL

#Model Accuracy, how often is the classifier correct?
print("Accuracy",metrics.accuracy_score(y_test,y_pred))

