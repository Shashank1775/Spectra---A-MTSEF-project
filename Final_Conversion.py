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
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from keras.utils import to_categorical

class CreateModels():
    def __init__(self, dataframe, column_Names, columns_to_exclude, class_column, save_path, prediction_data=None, drop_certain=False, drop_index=None, simple=True, replace=None, default=True):
        '''
        Purpose -- Initialize basic variables from the given database, expand database such that all test cases are even (SMOTE)
        Assumption -- The database has already been cleaned, all unnecessary information has been removed
        Note -- The program will check if NA values are present and replace them with the number 0
        Note -- The program will check for string values and attempt to convert them to numbers using an encoder
        Note -- When writing columns_to_exclude make sure to write the same name as written in column_Names, as if this is not true the
        function will return an error
        Note -- Drop index needs to be input as a list with two items, one the start of the drop and the other the end of the drop
        '''
        self.simple = simple
        self.LE = LabelEncoder()
        self.df = pd.read_csv(dataframe, names=column_Names, low_memory=False)
        self.smote = SMOTE(random_state=17)
        self.scaler = MinMaxScaler()
        self.df_TClass = None
        self.exclude = columns_to_exclude
        self.drop = drop_certain
        self.dropIndex = drop_index
        self.default = default
        self.replace = replace
        self.class_column = class_column
        self.save_path = save_path
        self.prediction_data = prediction_data
        self.counter = 0

    def DifDataChecks(self, pred_data=False):
        '''
        Purpose -- This function is responsible for the various checks on the data assuring that errors won't occur
        later in the program
        Note -- This function only checks base level things such as the presence of NA, presence of str's, and other minor details
        Note -- This function will try to fix the data to the best of its abilities, currently it can change binary str output to number,
        replace NA values, and replace str(letters) if they are non-binary(multiple categories)
        Note -- This will also remove the columns specified as wanting to exclude, thus is very important
        '''
        if counter != 0:
            return 0
        if pred_data == False:
            if len(self.exclude) > 0:
                for column in self.exclude:
                    try:
                        self.df.pop(column)
                    except KeyError:
                        print(f"Column '{column}' not found in the DataFrame.")

            try:
                if self.drop:
                    self.df.drop(self.df.index[self.dropIndex[0]:self.dropIndex[1]], inplace=True)
                else:
                    print('Nothing to drop')
            except Exception as e:
                print(f"An error occurred: {e}")

            column = self.df.columns

            for cols in column:
                print(self.class_column)
                if cols != str(self.class_column):

                    print(f'Starting: {cols}')
                    try:
                        # Try to convert the value to an integer
                        self.df[cols] = pd.to_numeric(self.df[cols], errors='raise')
                    except ValueError:
                        # Handle non-numeric values
                        values = {item: i + 1 for i, item in enumerate(self.df[cols].unique())}
                        self.df[cols] = self.df[cols].replace(values)
                        try:
                            # Attempt to convert the column to int64
                            self.df[cols] = self.df[cols].astype('int64')
                        except ValueError:
                            print(f"Column {cols} still contains non-numeric values after replacement.")

            return self.df
        else:
            column = self.prediction_data.columns

            for cols in column:
                try:
                    self.prediction_data[cols] = pd.to_numeric(self.prediction_data[cols], errors='raise')
                except ValueError:
                    values = {item: i + 1 for i, item in enumerate(self.df[cols].unique())}
                    self.prediction_data[cols] = self.prediction_data[cols].replace(values)
                    try:
                        self.prediction_data[cols] = self.prediction_data[cols].astype('int64')
                    except ValueError:
                        print(f"Column {cols} still contains non-numeric values after replacement.")
        self.counter += 1
    def Training_Test_Validation(self):
        '''
        Purpose -- Split data into train, test, split, so that the following models can be trained correctly
        Assumption -- Previous data clean up has occurred, dataframe and variables have been initialized
        Note -- In this step, a check will occur rating the quality of the data
        '''
        df = self.df.copy()
        if self.simple:
            print("Default")
            if self.default:
                Sub_To_Main = {'C': 'C', 'B': 'B', 'S': 'S', 'V': 'V', 'L': 'L', 'X': 'X', 'XC': 'X', 'TDG': 'T',
                               'K': 'K', 'C*': 'C',
                               'T': 'T', 'M': 'M', 'S*': 'S', 'P': 'P', 'XFC': 'X', 'SC*': 'S', 'DCX:': 'D', 'BU': 'B',
                               'F': 'F',
                               'BCU': 'B', 'A': 'A', 'CX:': 'C', 'D': 'D', 'FC': 'F', 'PC': 'P', 'SCTU': 'S', 'Ch': 'C',
                               'CX': 'C',
                               'CXF': 'C', 'CP': 'C', 'C:': 'C', 'CSGU': 'C', 'R': 'R', 'DP': 'D', 'DT': 'D', 'DX': 'D',
                               'Xe': 'X',
                               'DCX': 'D', 'F:': 'F', 'TD': 'T', 'CSU': 'C', 'CU': 'C', 'CB': 'C', 'FCX': 'F',
                               'CPF': 'C', 'P*': 'P',
                               'PD': 'P', 'CGU': 'C', 'DU:': 'D', 'XCU': 'X', 'CFB:': 'C', 'XDC': 'X', 'E': 'E',
                               'FCX:': 'F', 'SD': 'S',
                               'G': 'G', 'CD:': 'C', 'ST': 'S', 'CP:': 'C', 'DX:': 'D', 'CF': 'C', 'CGSU': 'C',
                               'FX:': 'F', 'SU': 'S',
                               'SC': 'S', 'MU': 'M', 'DXCU': 'D', 'XFU': 'X', 'GS:': 'G', 'CBU:': 'C', 'CTGU:': 'C',
                               'XF': 'X',
                               'XB': 'X',
                               'DTU': 'D', 'CB:': 'C', 'FC:': 'F', 'FU': 'F', 'FXU:': 'F', 'XD:': 'X', 'PF': 'P',
                               'CFU:': 'C',
                               'XSC': 'X',
                               'DTU:': 'D', 'Xk': 'X', 'CGTP:': 'C', 'FCU': 'F', 'MU:': 'M', 'PDC': 'P', 'Xt': 'X',
                               'GC': 'G',
                               'XU': 'X',
                               'D': 'D', 'DU': 'D', 'E': 'E', 'F': 'F', 'A': 'A', 'R': 'R', 'XD': 'X', 'P:': 'P',
                               'AS': 'A', 'G:': 'G',
                               'CBU': 'C', 'SE*': 'S', 'Q': 'Q', 'SFC': 'S', 'DSU:': 'D', 'CPD': 'C', 'CFXU': 'C',
                               'SG': 'S', 'EU': 'E'}

                df['Class'].replace(Sub_To_Main, inplace=True)
            else:
                df['Class'].replace(self.replace, inplace=True)

        for value, count in df['Class'].value_counts().items():
            if count <= 6:
                df = df[df['Class'] != value]

        df.dropna(inplace=True)
        self.y = self.LE.fit_transform(df.pop('Class'))
        self.x = df
        self.x = self.scaler.fit_transform(self.x)
        # Perform SMOTE resampling
        self.x, self.y = self.smote.fit_resample(self.x, self.y)
        # Split data into train, validation, and test sets
        X_train, X_val_test, y_train, y_val_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def KNearestNeighbors(self, return_Prediction=False, n_neighbors=5):
        self.DifDataChecks()
        x_train, x_val, x_test, y_train, y_val, y_test = self.Training_Test_Validation()
        print('Starting KNN')

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train, y_train)

        if return_Prediction:
            y_pred = knn.predict(self.prediction_data)
            y_pred = self.LE.inverse_transform(y_pred)
            print(pd.DataFrame(y_pred))
        y_pred = knn.predict(x_test)
        knn_accuracy = accuracy_score(y_test, y_pred)
        print(f'KNN_Accuracy: {knn_accuracy}')

    def Sequential(self, saving_link, plot=True, create=False):
        self.DifDataChecks()
        x_train, x_val, x_test, y_train, y_val, y_test = self.Training_Test_Validation()
        print(len(pd.DataFrame(y_test).value_counts()))
        print(len(pd.DataFrame(y_train).value_counts()))

        def Create_Sequential():
            num_classes = len(np.unique(y_train))  # Get the number of unique classes
            print(num_classes)
            model = keras.Sequential(name="my_sequential")
            model.add(layers.Dense(64, activation="relu", name="layer1"))
            model.add(layers.Dense(32, activation="relu", name="layer2"))
            model.add(layers.Dense(16, activation="relu", name="layer3"))
            model.add(layers.Dense(num_classes, name="layer4", activation="softmax"))  # Use num_classes here

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',  # Change loss function to handle integer labels
                          metrics=['accuracy', 'mse'])

            history = model.fit(x_train, y_train, epochs=16, batch_size=128,
                                validation_data=(x_val, y_val))
            model.save(
                r"C:\\Users\\shash\\PycharmProjects\\Asteroid_Comp_Understand\\.venv\\Finale_API\\TensorFlowSequentialModel")
            return history

        if create:
            FitModel = Create_Sequential()

        # Load the saved model
        model = tf.keras.models.load_model(
            r"C:\\Users\\shash\\PycharmProjects\\Asteroid_Comp_Understand\\.venv\\Finale_API\\TensorFlowSequentialModel")

        if create:
            loss, accuracy, mse = model.evaluate(x_test, y_test)
            print(f'L:{loss}, A:{accuracy}, MSE:{mse}')

        # Returning predictions:
        if self.prediction_data != None:
            self.DifDataChecks(pred_data=True)
            scaledPred = self.scaler.fit_transform(self.prediction_data)
            # Predict Values
            y_pred = model.predict(scaledPred)
            # Convert Values Back
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = self.LE.inverse_transform(y_pred)
            # Print Values
            print(pd.DataFrame(y_pred))

    def RandomForest(self):
        self.DifDataChecks()
        x_train, x_val, x_test, y_train, y_val, y_test = self.Training_Test_Validation()

        rf_model = RandomForestClassifier(n_estimators=100, random_state=17)
        rf_model.fit(x_train, y_train)
        predictions = rf_model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(accuracy)

        if self.prediction_data != None:
            self.DifDataChecks(pred_data=True)
            scaledPred = self.scaler.fit_transform(self.prediction_data)
            # Predict Values
            y_pred = rf_model.predict(scaledPred)
            y_pred = self.LE.inverse_transform(y_pred)

        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Set tick marks for classes
        classes = np.unique(y_test)
        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)

        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()