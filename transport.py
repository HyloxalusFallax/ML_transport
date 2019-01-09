import codecs
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier

def load_data(data_directory):
	data = []
	labels = []
	with codecs.open('transport_data.csv',encoding='utf-8') as file:
		next(file)
		for line in file:
			splitted = line.split(",")
			log = float(splitted[0])
			lat = float(splitted[1])
			request_ts = float(splitted[2])
			trans_ts = float(splitted[3])
			labels.append(splitted[4][0])
			data.append([float(log), float(lat), int(request_ts), int(trans_ts), 1])
	return labels, data

labels, data = load_data('./')

data = np.array(data)
labels = np.array(labels)

x_coord = [x[0] for x in data]
y_coord = [x[1] for x in data]
colors = list(map(lambda x: (1., 0., 0., 1.) if x == '0' else ((0., 1., 0., 1.) if x == '1' else ((0., 0., 1., 1.) if x == '2' else ((0., 0., 0., 1.) if x == '-' else (0., 1., 1., 1.)))), labels))
plt.scatter(x_coord, y_coord, c=colors)
plt.show()

unique_labels = set(['0', '1', '2'])
list_ul = ['0', '1', '2']

cleaned_data = []
cleaned_labels = []
test_data = []
test_labels = []
for i in range(len(labels)):
	if (labels[i] != '?'):
		new_datum = [data[i][0], data[i][1]]
		cleaned_data.append(data[i])
		cleaned_labels.append(labels[i])
	if (labels[i] == '?'):
		test_data.append(data[i])

cleaned_data = np.array(cleaned_data)
cleaned_labels = np.array(cleaned_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

def distance(a_log, a_lat, b_log, b_lat):
	return math.sqrt((a_log - b_log)**2 + (a_lat - b_lat)**2)

perm = cleaned_data[:, 0].argsort()
cleaned_data = cleaned_data[perm]
cleaned_labels = cleaned_labels[perm]
list_of_isolated = []
new_cleaned_labels = cleaned_labels
for i in range(len(cleaned_data)):
	if (i % 1000 == 0):
		print(i)
	if cleaned_labels[i] == '-':
		min_distance = 100000
		j_of_min = -1
		if (i - 500 > 0):
			start = i - 500
		else:
			start = 0
		if (i + 500 < len(cleaned_data)):
			finish = i + 500
		else:
			finish = len(cleaned_data)
		for j in range(start, finish):
			dist = distance(cleaned_data[i][0], cleaned_data[i][1], cleaned_data[j][0], cleaned_data[j][1])
			if ((cleaned_labels[j] in unique_labels) and (dist < min_distance)):
				min_distance = dist
				j_of_min = j
		if(min_distance > 1e-3):
			list_of_isolated.append(i)
		new_cleaned_labels[i] = cleaned_labels[j_of_min]
		cleaned_data[i][4] = 0
cleaned_labels = new_cleaned_labels
print(cleaned_data.shape)
cleaned_data = np.delete(cleaned_data, list_of_isolated, axis=0)
cleaned_labels = np.delete(cleaned_labels, list_of_isolated, axis=0)		
print(cleaned_data.shape)
		
x_coord = [x[0] for x in cleaned_data]
y_coord = [x[1] for x in cleaned_data]
colors = list(map(lambda x: (1., 0., 0., 1.) if x == '0' else ((0., 1., 0., 1.) if x == '1' else ((0., 0., 1., 1.) if x == '2' else ((0., 0., 0., 1.) if x == '-' else (0., 1., 1., 1.)))), cleaned_labels))
plt.scatter(x_coord, y_coord, c=colors)
plt.show()
	
augumented_data = []
augumented_labels = []
				
for i in range(len(cleaned_labels)):
	if (cleaned_labels[i] in unique_labels):
		augumented_labels.append(list_ul.index(cleaned_labels[i]))
		new_datum = [cleaned_data[i][0], cleaned_data[i][1]]
		augumented_data.append(cleaned_data[i])
		
augumented_data = np.array(augumented_data)
augumented_labels = np.array(augumented_labels)

original_data = []
original_labels = []

additional_data = []
additional_labels = []

for i in range(len(augumented_data)):
	if (augumented_data[i][4]):
		original_data.append(augumented_data[i])
		original_labels.append(augumented_labels[i])
	else:
		additional_data.append(augumented_data[i])
		additional_labels.append(augumented_labels[i])

original_data = np.array(original_data)
original_labels = np.array(original_labels)
additional_data = np.array(additional_data)
additional_labels = np.array(additional_labels)	
		
data_train, data_test, labels_train, labels_test = train_test_split(original_data, original_labels, test_size= 1)

from sklearn.utils import shuffle

data_train = np.concatenate((data_train, additional_data), axis=0)
labels_train = np.concatenate((labels_train, additional_labels), axis=0)

data_train, labels_train = shuffle(data_train, labels_train)

data_train = np.delete(data_train, [2,3,4], axis=1)
data_test = np.delete(data_test, [2,3,4], axis=1)
test_data = np.delete(test_data, [2,3,4], axis=1)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
test_data = scaler.transform(test_data)

#clf = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, solver='adam', verbose=1, tol=0.0000000001, n_iter_no_change=50, alpha=0) #alpha=0.0000001,batch_size=32
#clf = RandomForestClassifier(n_estimators=500, n_jobs=4, min_samples_split=2, min_samples_leaf=1, max_features=None)
#clf = ExtraTreesClassifier(n_estimators=500, n_jobs=4)
#clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1)
clf = RandomForestClassifier(n_estimators=500, n_jobs=4, min_samples_split=2, min_samples_leaf=1, max_features=None)
clf.fit(data_train, labels_train)
y_pred = clf.predict(data_train)
print('train accuracy: ' + str(accuracy_score(labels_train, y_pred)))
y_pred = clf.predict(data_test)
print('test accuracy: ' + str(accuracy_score(labels_test, y_pred)))
y_pred = clf.predict(test_data)
print(y_pred)
x_coord = [x[0] for x in test_data]
y_coord = [x[1] for x in test_data]
colors = list(map(lambda x: (1., 0., 0., 1.) if x == 0 else ((0., 1., 0., 1.) if x == 1 else ((0., 0., 1., 1.) if x == 2 else ((0., 0., 0., 1.) if x == '-' else (0., 1., 1., 1.)))), y_pred))

plt.scatter(x_coord, y_coord, c=colors)
plt.show()

with codecs.open('output.txt','w',encoding='utf8') as f:
	for e in y_pred:
		f.write(str(e) + '\n')