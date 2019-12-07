#libraries used
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

# to read the file and separate the fields by semicolons
df = pd.read_csv("cardio_train.csv", sep = ";")

# to build the dataframe with the attributes we need
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

# add additional column called obesity
df["obesity"] = df.BMI.apply(obesity)

#User interface - menu
print("\n      Welcome to CorHealth      ")
print("\nWhat would you like to do today?")
print("\n", "                Menu                  ")
print("============================================")
print("Feature 1: Heart Disease Distribution")
print("Feature 2: Attribute Correlation to Cardiovascular Disease")
print("Feature 3: Key Attributes Analaysis")
print("Feature 4: Prediction", "\n")


while True: 
    print("\n                  Options                 ")
    print("==========================================")
    print("If you want Feature 1, input 1")
    print("If you want Feature 2, input 2")
    print("If you want Feature 3, input 3")
    print("If you want Feature 4, input 4")
    print("If you do not want to continue, input exit")
    
    selectFeature = input("Which feature do you want to see? ")
   
    if selectFeature == "1":
        """#Feature 1"""
        print("\nFeature 1")
        nocardio = len(df[df.cardio == 0])
        yescardio = len(df[df.cardio == 1])
        print("The % of patients who have not had cardio disease: {:.2f}%".format((nocardio / (len(df.cardio))*100)))
        print("The % of patients who have had cardio disease: {:.2f}%".format((yescardio / (len(df.cardio))*100)))
        
        fig, ax = plt.subplots()
        ax.set_title("Heart Disease Distribution")
        sns.countplot(x="cardio", data=df, palette="bwr")
        plt.xlabel("Cardio (1=Have disease, 0=No disease)")
        plt.show()
        print("")

    elif selectFeature == "2":
        """#Feature 2"""
        print("\nFeature 2")
        
        #set cardio as target variable
        y = df["cardio"]
        
        #attributes to be used 
        x = df[["age", "gender", "height", "weight", "cholesterol", "gluc", "smoke", "alco", "active", "obesity"]]
        
        
        logit_model=sm.Logit(y,x)
        result=logit_model.fit(disp = False)
        result.columns= ['category','percent']        
        gf = result.params
    
        #show the graph by correlation

        fig, ax = plt.subplots()
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Attributes')
        ax.set_title('Attributes vs Correlations')
        gf.plot(kind='bar',x='category',y='percent')
        plt.show()
    
        # show the users how much each attribute weights
        
        gf = gf.reset_index()
        gf.columns = ["Attribute", "Coefficient"]
        print(gf)
        
    elif selectFeature == "3":  
        """#Feature 3"""
        print("\nFeature 3")
       
        ff = df[df["gender"] == 1]
        mf = df[df["gender"] == 2]
         
        
        #plot graphs for selected features
        fig, ax = plt.subplots()
        ax.set_title("Gender vs Cardio")
        sns.countplot(x='gender', data=df, hue='cardio')
        plt.show()
        
        fig, ax = plt.subplots()
        ax.set_title("Cholesterol vs Cardio: Female")
        sns.countplot(x='cholesterol', hue='cardio', data=ff)
        plt.show()
        
        fig, ax = plt.subplots()
        ax.set_title("Cholesterol vs Cardio: Male")
        sns.countplot(x='cholesterol', hue='cardio', data=mf)
        plt.show()
        
        fig, ax = plt.subplots()
        ax.set_title("Active vs Cardio: Female")
        sns.countplot(x='active', hue='cardio', data=ff)
        plt.show()

        fig, ax = plt.subplots()
        ax.set_title("Active vs Cardio: Male")
        sns.countplot(x='active', hue='cardio', data=mf)
        plt.show()
    
    elif selectFeature == "4":   
        """#Feature 4"""
        print("\nFeature 4")
        #To separate the data into 75% of train data and 25% of test data
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
        lr = LogisticRegression()
        model = lr.fit(x_train, y_train)
        
        #predict the model
        y_pred = model.predict(x_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        #print(cnf_matrix)
        
        #find the accuracy of the model created
        accuracy = metrics.accuracy_score(y_test, y_pred)
        #print(accuracy)
        
        prob = model.predict_proba(x_test)
        #print(prob)
        
        #ask for user inputs
        age = input("What is patient's age? ")
        gen = input("What is patient's gender? (F/M) ")
        height = input("What is patient's height? (cm) ")
        weight = input("What is patient's weight? (kg) ")
        cholesterol = input("What is patient's cholesterol level? ( 1 = low, 2 = medium, 3 = high) ")
        glucose = input("What is patient's glucose level? ( 1 = low, 2 = medium, 3 = high) ")
        smoke = input("Does the patient smoke? (Yes = 1, No = 0) ")
        alcohol = input("Does the patient drink alcohol? (Yes = 1, No = 0) ")
        active = input("Does the patient work out? (Yes = 1, No = 0) ")
        obesity = round((int(weight) / ((int(height) / 100) ** 2)))
        
        #Convert gender input into numbers
        if gen == "F":
            gender = 1
        elif gen == "M":
            gender = 2
        
        inp = [[age, gender,height, weight, cholesterol, glucose,smoke,alcohol,active,obesity]]
        pin = pd.DataFrame(inp, columns = ["age", "gender", "height", "weight", "cholesterol", "gluc", "smoke", "alco", "active", "obesity"])
        
        #Based on user inputs, predict the percentage of the person getting heart disease 
        pred = model.predict(pin)
        prob = int(model.predict_proba(pin)[:,1] * 100)
        print("\n")
        #Output the result
        print("The patient has", str(prob)+"% chance of getting cardio disease.")
        
        
    else:
        print("\nThank you for using our software! We hope to see you again.")
        print("\1")
        break 


    
    
