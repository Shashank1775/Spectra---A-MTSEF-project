import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib

cols = ['Number','Name','Desig','Family','CSource',
        'Class','DiamSource','DiamFlag','Diam',
        'HSource','H','HBand','GSource','G','G2','AlbSource'
    ,'AlbFlag','Albedo','PFlag','Period','PerDesc','AmpFlag'
    ,'AmpMin','AmpMax','U','Notes','IsBinary','Pole','Survey'
    ,'ExNotes','Private']
df = pd.read_csv(r'C:\Users\shash\PycharmProjects\Asteroid_Comp_Understand\.venv\Finale_API\LC_Summary\lc_summary.csv', names = cols, low_memory=False, na_values="-")
#Preparing Data - Cleaning
df.drop(df.index[:22], inplace=True)
print(type(df['Number'].head()))
#Removing Letter Featurers that don't matter overall
Names = df.pop('Name')
Designation = df.pop('Desig')
Dim_Source = df.pop('DiamSource')
Dim_Flag = df.pop('DiamFlag')
H_Band = df.pop('HBand')
G_Source = df.pop('GSource')
Alb_Source = df.pop('AlbSource')
Alb_Flag = df.pop('AlbFlag')
P_Flag = df.pop('PFlag')
Amp_Flag = df.pop('AmpFlag')
Survey = df.pop('Survey')
Private = df.pop('Private')
df['Notes'].fillna('0', inplace=True)
Notes = df.pop('Notes')
HSource = df.pop('HSource')
PerDesc = df.pop('PerDesc')
#Changing Letters to numbers, important features
df['IsBinary'].fillna('?', inplace=True)
df['U'].fillna('?', inplace=True)
#creating replacement function
def replace(rep_array, column):
    return df[column].replace(rep_array)
#creating replacement arrays
U_Values = {'3-':-3.0,'2+':2.5,'1+':1.5,'2-':-2, '1-':-1, '?':0}
CSource_Values = {'S':1,'T':2,'A':3,'L':4,'M':5, '':0 }
IsBinary_Values = {'M':1,'B':2,'?':0}
#using replacement function
df['CSource'] = replace(CSource_Values, 'CSource')
df['IsBinary'] = replace(IsBinary_Values, 'IsBinary')
df['U'] = replace(U_Values, 'U')
#Chaning binary values to be either 1 or 0
df['Pole'] = (df['Pole'] == 'Y').astype(int)
df['ExNotes'] = (df['ExNotes'] == 'Y').astype(int)
#removing rows that don't have family assigned ~ can be used as test data later
rows_with_nan_in_family = df[df['Family'].isnull()]
nan_family_rows_for_reference = rows_with_nan_in_family.copy()
df = df.dropna(subset=['Family'])
#removing rows that don't have class assigned
rows_with_nan_in_class = df[df['Class'].isnull()]
nan_class_rows_for_reference = rows_with_nan_in_class.copy()
df = df.dropna(subset=['Class'])
print(df['Family'])
Fam = df.pop('Family')
'''
new = []
for i in df['CSource']:
    if i not in new:
        new.append(i)

print(new)
#print(df.shape)
Dif_Family = df['Family'].unique()
Dif_Family = Dif_Family[:-1]
counts = df['Family'].value_counts()

plt.figure(figsize=(40, 6))  # Adjust figure size if needed
plt.bar(range(len(Dif_Family)), counts)
plt.xlabel('Family')
plt.ylabel('Count')
plt.xticks(range(len(Dif_Family)), Dif_Family, rotation=90)  # Set x-axis labels
plt.tight_layout()
plt.show()
binary = df['IsBinary'].unique()
counts = df['IsBinary'].value_counts()
plt.figure(figsize=(10,6))
plt.bar(range(len(binary)), counts)
plt.xlabel('Binary')
plt.ylabel('Count')
plt.xticks(range(len(binary)),binary,rotation = 45)
plt.show()
'''
#Combining sub-classes into their main classes, good to shorten classification range
print('Done')
Sub_To_Main = {'C': 'C', 'B': 'B', 'S': 'S', 'V': 'V', 'L': 'L', 'X': 'X', 'XC': 'X', 'TDG': 'T', 'K': 'K', 'C*': 'C',
               'T': 'T', 'M': 'M', 'S*': 'S', 'P': 'P', 'XFC': 'X', 'SC*': 'S', 'DCX:': 'D', 'BU': 'B', 'F': 'F',
               'BCU': 'B', 'A': 'A', 'CX:': 'C', 'D': 'D', 'FC': 'F', 'PC': 'P', 'SCTU': 'S', 'Ch': 'C', 'CX': 'C',
               'CXF': 'C', 'CP': 'C', 'C:': 'C', 'CSGU': 'C', 'R': 'R', 'DP': 'D', 'DT': 'D', 'DX': 'D', 'Xe': 'X',
               'DCX': 'D', 'F:': 'F', 'TD': 'T', 'CSU': 'C', 'CU': 'C', 'CB': 'C', 'FCX': 'F', 'CPF': 'C', 'P*': 'P',
               'PD': 'P', 'CGU': 'C', 'DU:': 'D', 'XCU': 'X', 'CFB:': 'C', 'XDC': 'X', 'E': 'E', 'FCX:': 'F', 'SD': 'S',
               'G': 'G', 'CD:': 'C', 'ST': 'S', 'CP:': 'C', 'DX:': 'D', 'CF': 'C', 'CGSU': 'C', 'FX:': 'F', 'SU': 'S',
               'SC': 'S', 'MU': 'M', 'DXCU': 'D', 'XFU': 'X', 'GS:': 'G', 'CBU:': 'C', 'CTGU:': 'C', 'XF': 'X',
               'XB': 'X',
               'DTU': 'D', 'CB:': 'C', 'FC:': 'F', 'FU': 'F', 'FXU:': 'F', 'XD:': 'X', 'PF': 'P', 'CFU:': 'C',
               'XSC': 'X',
               'DTU:': 'D', 'Xk': 'X', 'CGTP:': 'C', 'FCU': 'F', 'MU:': 'M', 'PDC': 'P', 'Xt': 'X', 'GC': 'G',
               'XU': 'X', }
df['Class'] = df['Class'].replace(Sub_To_Main)
Sub_To_Main.update({
    'D': 'D', 'DU': 'D', 'E': 'E', 'F': 'F', 'A': 'A', 'R': 'R', 'XD': 'X', 'P:': 'P', 'AS': 'A', 'G:': 'G',
    'CBU': 'C', 'SE*': 'S', 'Q': 'Q', 'SFC': 'S', 'DSU:': 'D', 'CPD': 'C', 'CFXU': 'C', 'SG': 'S', 'EU': 'E',
    'SDU::': 'S', 'O': 'O', 'Sq': 'S', 'CZ': 'C', 'Xn': 'X', 'Xc': 'X', 'Sr': 'S', 'Xe*': 'X', 'Cgh': 'C'
})
df['Class'] = df['Class'].replace(Sub_To_Main)
Class = df['Class'].copy()
#More Pre-Processing
from sklearn.preprocessing import LabelEncoder

values_to_remove = ['O','R','Q','G']

for remove in values_to_remove:
    df = df.drop(df[df['Class'] == remove].index)
#Creating Y for testing :

#Splitting Data
for columns in df.columns:
    if df[columns].dtype == 'float64':
        df[columns] = df[columns].astype('int64')
print('Done')
def split_data(X, y):
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
def SMOTE1(x,y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    smote = SMOTE(random_state=17)
    X_resampled, y_resampled_encoded = smote.fit_resample(x, y)
    return X_resampled, y_resampled_encoded
def Scale(x):
    scaler = MinMaxScaler()
    X_resampled = scaler.fit_transform(x)
    return X_resampled

y = df.pop('Class')
x = df
X, y = SMOTE1(x, y)
X = Scale(X)
print('Done')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
#print(y_val.value_counts())
#Model Training
'''
#MAKING GRAPH
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
plt.bar(y_val.unique(), y_val.value_counts(), color='skyblue')

# Add labels and title
plt.xlabel('Classes')
plt.ylabel('Amount(int)')
plt.title('Class Spread in Training Data')

# Show plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better fit
plt.show()
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score
from keras.utils import to_categorical


def calculate_knn_accuracy(X_train, X_test, y_train, y_test, n_neighbors=5):
    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier using the training data
    knn.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    knn_accuracy = accuracy_score(y_test, y_pred)

    return knn_accuracy
print('Done')
#accuracy = calculate_knn_accuracy(X_train, X_test, y_train, y_test)
#print(accuracy)

'''

# Plotting Loss vs Accuracy
plt.figure(figsize=(8, 6))
plt.plot(accuracy, loss, label='Training')
plt.plot(val_accuracy, val_loss, label='Validation')
plt.title('Loss vs Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Create pie chart
plt.figure(figsize=(8, 6))  # Set the figure size
plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=140)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Class Distribution after SMOTE')  # Set title

plt.show()  # Display the pie chart
'''
def preprocess(y,astype=True,x_change=False):
    if x_change:
        X = y
        X = np.asarray(X).astype('float32')
        return X
    else:
        y = pd.get_dummies(y, columns=['Class'])
        if astype:
            y = y.astype(int)
        return y

x = preprocess(X_train, x_change=True)
pd.DataFrame(x)
y = preprocess(y_train, True)
xval = preprocess(X_val, x_change=True)
yval = preprocess(y_val)
xtest = preprocess(X_test, x_change=True)
ytest = preprocess(y_test, )
print('Done')
def CreateSequential(x,y,xval, yval):
    model = keras.Sequential(name="my_sequential")
    model.add(layers.Dense(64, activation="relu", name="layer1"))
    model.add(layers.Dense(32, activation="relu", name="layer2"))
    model.add(layers.Dense(16, activation="relu", name="layer3"))
    model.add(layers.Dense(14, name="layer4", activation="softmax"))

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'mse'])

    FitModel = model.fit(x, y, epochs=16, batch_size=128, validation_data=(xval, yval))

    test_loss, test_accuracy, test_mse = model.evaluate(xtest, ytest)
    y_pred = model.predict(xtest)
    print(y_pred)
    print(f'TL:{test_loss}, TA:{test_accuracy}, TMSE:{test_mse}')



    train_acc = FitModel.history['accuracy']
    val_acc = FitModel.history['val_accuracy']

    # Plotting accuracy values over epochs
    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model.save("TensorFlowSequentialModel")
CreateSequential(x,y, xval,yval)
#RFM(Random Forest Model)
print('Starting RFC')
def Rf_Model():
    rf_model = RandomForestClassifier(n_estimators=100, random_state=17)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)

