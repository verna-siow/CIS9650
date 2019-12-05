import pandas as pd
from sklearn import metrics # Confusion matrix (Feature 3)
from sklearn.linear_model import LogisticRegression # prediction model(Feature 3)
import seaborn as sns # graphs (Feature 2)
sns.set()
from sklearn.model_selection import train_test_split #train and split raw dataset (Feature 3)
import statsmodels.api as sm # stats analysis (Feature 1)
from warnings import simplefilter #remove warnings sign 
import matplotlib.pyplot as plt

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv("cardio_train.csv", sep = ";")

df = df[["age", "gender", "height", "weight", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]]

#convert age in days to years
df["age"] = (df["age"] / 365).astype(int)

#set up a column for BMI using the BMI formula in the internet
df["BMI"] = df.apply(lambda row: round((row["weight"] / ((row["height"] / 100) ** 2)), 1), axis = 1)

#using the BMI to determine if the person is obesed or not
def obesity(x):
    if x >= 30:
        return 1
    else:
        return 0

df["obesity"] = df.BMI.apply(obesity)
print("      Welcome to CorHealth      ")
print("   What would you like to do today?     ")
print("\n", "                Menu                  ")
print("============================================")
print("Feature 1: Attribute Correlation to Cardiovascular Disease")
print("Feature 2: Key Attributes Analaysis")
print("Feature 3: Prediction", "\n")


while True: 
    print("                  Options                 ")
    print("==========================================")
    print("If you want Feature 1, input 1")
    print("If you want Feature 2, input 2")
    print("If you want Feature 3, input 3")
    print("If you do not want to continue, input exit")
    
    selectFeature = input("Which feature do you want to see? ")
    if selectFeature == "1":
        """#Feature 1"""
        print("Feature 1")
        
        y = df["cardio"]
        
        x = df[["age", "gender", "height", "weight", "cholesterol", "gluc", "smoke", "alco", "active", "obesity"]]
        
        
        logit_model=sm.Logit(y,x)
        result=logit_model.fit(disp = False)
        
        gf = result.params
        print(gf)
        result.columns=['category','percent']
    
        
        gf.plot(kind='bar',x='category',y='percent')
        plt.show()
    
    elif selectFeature == "2":  
        """#Feature 2"""
        print("Feature 2")
        
        ff = df[df["gender"] == 1]
        mf = df[df["gender"] == 2]
        
        print("Gender vs Cardio")
        sns.countplot(x='gender', data=df, hue='cardio')
        plt.show()
        
        print("Cholesterol vs Cardio: Female ")
        sns.countplot(x='cholesterol', hue='cardio', data=ff)
        plt.show()
        
        print("Cholesterol vs Cardio: Male ")
        sns.countplot(x='cholesterol', hue='cardio', data=mf)
        plt.show()
        
        print("Active vs Cardio: Female")
        sns.countplot(x='active', hue='cardio', data=ff)
        plt.show()
        
        print("Active vs Cardio: Male")
        sns.countplot(x='active', hue='cardio', data=mf)
        plt.show()
    
    elif selectFeature == "3":   
        """#Feature 3"""
        print("Feature 3")
        #To separate the data into 75% of train data and 25% of test data
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
        lr = LogisticRegression()
        model = lr.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        #print(cnf_matrix)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        #print(accuracy)
        
        prob = model.predict_proba(x_test)
        #print(prob)
        
        age = input("What is your age? ")
        gender = input("What is your gender? ( F = 1, M = 2) ")
        height = input("What is your height? (cm) ")
        weight = input("What is your weight? (kg) ")
        cholesterol = input("What is your cholesterol level? ( 1 = low, 2 = medium, 3 = high) ")
        glucose = input("What is your glucose level? ( 1 = low, 2 = medium, 3 = high) ")
        smoke = input("Do you smoke? (Yes = 1, No = 0) ")
        alcohol = input("Do you drink alcohol? (Yes = 1, No = 0) ")
        active = input("Do you work out? (Yes = 1, No = 0) ")
        obesity = round((int(weight) / ((int(height) / 100) ** 2)))
        
        
        inp = [[age, gender,height, weight, cholesterol, glucose,smoke,alcohol,active,obesity]]
        pin = pd.DataFrame(inp, columns = ["age", "gender", "height", "weight", "cholesterol", "gluc", "smoke", "alco", "active", "obesity"])
        
        pred = model.predict(pin)
        print("\n")
        if pred == 0: 
            print("You do not have a chance of getting cardio disease")
        else: 
            print("You have a chance of getting cardio disease")
        
    else:
        print("Thank you for using our software! We hope to see you again.")
        print("\1")
        break 


    
    
