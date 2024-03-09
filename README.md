Description:

*DISCLAIMER* - THIS IS A WORK IN PROGRESS MORE ITEMS WILL BE ADDED 

This Python script encapsulates a versatile class, CreateModels, tailored for streamlined data preprocessing, model creation, and evaluation. It serves as a comprehensive toolkit for machine learning tasks, particularly classification, leveraging popular libraries such as TensorFlow, Keras, and scikit-learn.

Key Features:

Data Preprocessing: The DifDataChecks method efficiently handles various data preprocessing tasks, including handling missing values, encoding categorical variables, and excluding specified columns. It ensures data integrity and prepares it for model training.

Example:

python
Copy code
model = CreateModels(dataframe='data.csv', column_Names=['col1', 'col2'], columns_to_exclude=['col3'], class_column='target', save_path='saved_models')
preprocessed_data = model.DifDataChecks()
Model Creation: The script offers functionalities to create three distinct types of classification models: K-Nearest Neighbors (KNN), Random Forest, and Sequential Neural Networks. Each model is implemented using appropriate libraries, enabling flexibility and scalability.

K-Nearest Neighbors (KNN):

python
Copy code
model.KNearestNeighbors(return_Prediction=True, n_neighbors=5)
Random Forest:

python
Copy code
model.RandomForest()
Sequential Neural Networks:

python
Copy code
model.Sequential(saving_link='model.h5', plot=True, create=True)
Model Evaluation: After model training, the script provides evaluation metrics such as accuracy scores and confusion matrices to assess model performance. It includes visualization capabilities to facilitate easy interpretation of results.

Customization and Flexibility: The class allows for customization of model parameters, data handling strategies, and output options, catering to diverse machine learning tasks and datasets.

Compatibility and Documentation: The script is designed to be compatible with various data formats and environments. Detailed comments within the code enhance readability and facilitate understanding for users.

Usage:

Users can simply instantiate the CreateModels class with their dataset and desired configurations.
Methods such as KNearestNeighbors, RandomForest, and Sequential can be called to create and evaluate specific models.
Customization options enable users to tailor the workflow according to their specific requirements and preferences.
By encapsulating essential functionalities within a single class, this script aims to simplify the process of data preprocessing, model creation, and evaluation, thereby empowering users to efficiently tackle classification tasks in machine learning projects.
