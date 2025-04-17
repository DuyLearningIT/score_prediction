# THUẬT TOÁN KNN
import math
import pandas as pd 
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

usecols = ["Gioi tinh", "Gio hoc", "So buoi hoc them", "Tong diem"]

data = pd.read_csv('C:\\Learning_IT\\Python\\ML_CORE\\HỌC ML\\DATA\\du_lieu_hoc_sinh.csv', encoding='latin1', usecols= usecols)

df = pd.DataFrame(data)

# Chuẩn hóa dữ liệu: Nam = 1, Nữ = 0, Giảm các trường còn lại 0 <= x <= 1
# df.iloc để lấy cả 1 cái obj

df["Gioi tinh"] = df["Gioi tinh"].apply(lambda x: 1 if x == "Nam" else 0)
df["Gio hoc"] = df["Gio hoc"].apply(lambda x: (x / max(df["Gio hoc"])))
df["So buoi hoc them"] = df["So buoi hoc them"].apply(lambda x: (x / max(df["So buoi hoc them"])))

df_in = df[["Gioi tinh","Gio hoc","So buoi hoc them"]]
df_out = df[["Tong diem"]]

# Chuyển object thành vector 
def ObjectToVector(obj):
	vector = np.array([i for i in obj])
	return vector 

# Tính khoảng cách giữa 2 vector 
def DistanceVec(vec1, vec2):
	dis = np.linalg.norm(np.array(vec1) - np.array(vec2))
	return dis

# Tính trọng số khoảng cách
def Weight(vec1, vec2, sigma = 1):
	return np.exp( (- DistanceVec(vec1, vec2)** 2) / sigma)

# Dự đoán điểm dựa trên 5 thằng gần nhất 5-NN
def KNN(vec):

	arrVec = np.array([ObjectToVector(vector) for vector in df_in.iloc])
	arrDis = np.array([DistanceVec(vec, vector) for vector in arrVec])
	sorted_index = sorted(range(len(arrDis)), key = lambda i : arrDis[i])
	top5_index = sorted_index[:5]
	
	tu = 0
	mau = 0
	for i in top5_index:
		tu += df_out["Tong diem"][i] * Weight(vec, arrVec[i])
		mau += Weight(vec, arrVec[i])

	predicted_score = tu / mau
	return predicted_score

print(KNN([1, 9/13, 1/4]))