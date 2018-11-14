import pandas as pd
import matplotlib.pyplot as plt

#from sklearn import datasets
#columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
#diabetes = datasets.load_diabetes()
#df = pd.DataFrame(diabetes.data, columns=columns)
#y = diabetes.target
#df = pd.read_csv('indians-diabetes.csv', sep=',', names=columns) # load date from file

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names=names)
# print(df)
# Pregnancies - Number of times pregnant - Numeric
# Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test - Numeric
# BloodPressure - Diastolic blood pressure (mm Hg) - Numeric
# SkinThickness - Triceps skin fold thickness (mm) - Numeric
# Insulin - 2-Hour serum insulin (mu U/ml) - Numeric
# BMI - Body mass index (weight in kg/(height in m)^2) - Numeric
# DiabetesPedigreeFunction - Diabetes pedigree function - Numeric
# Age - Age (years) - Numeric
# Outcome - Class variable (0 or 1) - Numeric
#df.boxplot()
#df.hist()
#df.groupby('class').hist()
#for i in names:
df.groupby('class').plas.hist(alpha=0.4) #alpha - prozra4nost
plt.show()