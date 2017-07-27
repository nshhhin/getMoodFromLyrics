from sklearn import svm,datasets
import numpy as np

# x,y座標を第1～4象限に分類する
data = np.array([[1,1],[-1,1],[-1,-1],[1,-1]]) # x,y座標の組
target = np.array([1,2,3,4])                   # 分類先 第1～4象限
print(data)
print(target)

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(data,target) # 学習

p_data = np.array([[10,10]])  # x,y = (10,10)
c = clf.predict(p_data) # 予測
print(c) # 第一象限