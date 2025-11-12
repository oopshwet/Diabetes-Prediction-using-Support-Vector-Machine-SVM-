🩺 Diabetes Prediction using Support Vector Machine (SVM)
This project implements a Machine Learning model to predict whether a person has diabetes based on diagnostic measurements. The model uses the PIMA Indian Diabetes Dataset and applies a Support Vector Machine (SVM) classifier with a linear kernel for classification.
📊 Dataset
The dataset used is the PIMA Diabetes Dataset, which contains several medical predictor variables and one target variable (Outcome).
Feature	Description
Pregnancies	Number of times pregnant
Glucose	Plasma glucose concentration (mg/dL)
BloodPressure	Diastolic blood pressure (mm Hg)
SkinThickness	Triceps skin fold thickness (mm)
Insulin	2-Hour serum insulin (mu U/ml)
BMI	Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction	A function that scores likelihood of diabetes based on family history
Age	Age of the person (years)
Outcome	1 = Diabetic, 0 = Non-Diabetic
🧠 Project Workflow
1. Data Collection and Analysis
The dataset is loaded using Pandas:
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
Data is inspected using:
diabetes_dataset.head()
2. Splitting Features and Target
Features (X) and target (Y) are separated:
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
3. Train-Test Split
The dataset is split into training and test data (80%-20%):
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
Shapes of the datasets:
X: (768, 8)
X_train: (614, 8)
X_test: (154, 8)
4. Model Training
An SVM classifier with a linear kernel is used:
from sklearn import svm

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
5. Model Evaluation
Accuracy on training data:
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
Output:
Accuracy score of the training data :  0.7866
Accuracy on test data:
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
📈 Results
Dataset	Accuracy
Training	78.66%
Testing	(To be calculated from the model run)
The model performs reasonably well for a linear classifier, suggesting that some features may be linearly separable for diabetic vs. non-diabetic classification.
🛠️ Technologies Used
Python 3
NumPy — for numerical computations
Pandas — for data manipulation
Scikit-learn — for ML model building and evaluation
Google Colab / Jupyter Notebook — for implementation
📁 Project Structure
Diabetes_Prediction_SVM/
│
├── diabetes.csv               # Dataset file
├── diabetes_prediction.ipynb  # Notebook file
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies (optional)
⚙️ Installation and Usage
Clone this repository:
git clone https://github.com/your-username/Diabetes_Prediction_SVM.git
cd Diabetes_Prediction_SVM
Install dependencies:
pip install numpy pandas scikit-learn
Run the notebook:
jupyter notebook diabetes_prediction.ipynb
Upload diabetes.csv into your /content/ folder (if using Google Colab).
🚀 Future Improvements
Apply data normalization or standardization to improve accuracy.
Try different kernels (RBF, polynomial) in SVM.
Compare results with other classifiers like:
Logistic Regression
Random Forest
K-Nearest Neighbors
Perform hyperparameter tuning using GridSearchCV.
