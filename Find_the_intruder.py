import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def sklearn_KMeans(k):
	# x-axis label 
	plt.xlabel(data_set.columns[0]) 
	# y-axis label
	plt.ylabel(data_set.columns[1]) 
	# plot title 
	plt.title('DDOS attack filter')
	kmeans = KMeans(k)
	kmeans.fit(data_set)
	print(kmeans.cluster_centers_)
	plt.scatter(data_set.iloc[:,0],data_set.iloc[:,1],c=kmeans.labels_,cmap='rainbow')  
	for index_centroids in range(k):
		plt.scatter(*(kmeans.cluster_centers_[index_centroids]),color="black")  
	plt.show()
	
def eucldien(point,centroid):
	return np.sqrt(sum((np.array(point)-np.array(centroid))**2))
class my_k_means:
	 def __init__(self, k,iteration=100):
	 	self.k = k
	 	self.iteration=iteration
	 	

	 def fit(self,data):
	 	data_set=data
	 	
	 	num_of_datapoints,num_of_features=data_set.shape
	 	features=[]
	 	for index_of_features in range(num_of_features):
	 		features.append(data_set.iloc[:,index_of_features])
	 	centroids=[]
	 	centroid_index=rnd.sample([i for i in range(0,num_of_datapoints)],self.k)
	 	for index_centroids in range(self.k):
	 		centroids.append(list(data_set.iloc[centroid_index[index_centroids]]))
	 	for iteration in range(self.iteration):
	 		labels=[]
	 		dist=np.zeros(self.k)
	 		clusters={}
	 		for index_data_points in range(num_of_datapoints):
	 			for index_centroids in range(self.k):
	 				dist[index_centroids]=eucldien([features[0][index_data_points],features[1][index_data_points]],centroids[index_centroids])
	 			min_dist=np.argmin(dist)
	 			if min_dist in clusters:
	 				clusters[min_dist].append([features[0][index_data_points],features[1][index_data_points]])
	 			else:
	 				clusters[min_dist]=[[features[0][index_data_points],features[1][index_data_points]]]
	 			labels.append(min_dist)
	 		plt.xlabel(data_set.columns[0])
	 		plt.ylabel(data_set.columns[1])
	 		plt.title('DDOS attack filter')
	 		if(len(clusters.keys())!=self.k):
	 			print("cluster  removed")
	 			plt.scatter(features[0],features[1],c=labels,cmap='rainbow')
	 			for index_centroids in range(self.k):
	 				plt.scatter(*(centroids[index_centroids]),color="black")
	 				plt.show()
	 			break
	 		new_centroids=[]
	 		for index_centroids in range(self.k):
	 			sum_x,sum_y=0,0
	 			for x,y in clusters[index_centroids]:
	 				sum_x+=x
	 				sum_y+=y
	 			new_centroids.append([(sum_x/num_of_datapoints),(sum_y/num_of_datapoints)])
	 		if(new_centroids==centroids):
	 			self.cluster_centers_=centroids
	 			self.labels_=labels
	 			plt.scatter(features[0],features[1],c=labels,cmap='rainbow')
	 			for index_centroids in range(self.k):
	 				plt.scatter(*(centroids[index_centroids]),color="black")
	 			plt.show()
	 			break
	 		else:
	 			centroids=new_centroids
	 		if iteration==(self.iteration-1):
	 			self.cluster_centers_=centroids
	 			self.labels_=labels
	 			plt.scatter(features[0],features[1],c=labels,cmap='rainbow')
	 			for index_centroids in range(self.k):
	 				plt.scatter(*(centroids[index_centroids]),color="black")
	 			plt.show()
	 			break

	 def predict(self,point):
	 	dist=np.zeros(self.k)
	 	for index_centroids in range(self.k):
	 		dist[index_centroids]=eucldien(np.array(point),self.cluster_centers_[index_centroids])
	 	min_dist=np.argmin(dist)
	 	return(min_dist)


data_set=pd.read_csv('DDOSdataset.csv','\t')
my_model=my_k_means(2)
my_model.fit(data_set)
print(my_model.cluster_centers_)
print(my_model.labels_)
print(my_model.predict([1.5, 1.5]))
sklearn_KMeans(2)