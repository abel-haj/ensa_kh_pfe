import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings



data = pd.read_csv(r'./creditcard.csv')
df = data.copy() # To keep the data as backup
df.head()



#df.shape
df = df.drop(labels='Time', axis=1)
df



df.isnull().sum()



df.dtypes



df.describe()



df.Class.value_counts()



sns.countplot(x=df.Class, hue=df.Class)



plt.figure(figsize=(10, 5))
sns.distplot(df.Amount)



df['Amount-Bins'] = ''



def make_bins(predictor, size=50):
    '''
    Takes the predictor (a series or a dataframe of single predictor) and size of bins
    Returns bins and bin labels
    '''
    bins = np.linspace(predictor.min(), predictor.max(), num=size) #construction d'un vecteur de taille 50 elements

    bin_labels = []

    # Index of the final element in bins list
    bins_last_index = bins.shape[0] - 1

    for id, val in enumerate(bins):
        if id == bins_last_index:
            continue
        val_to_put = str(int(bins[id])) + ' to ' + str(int(bins[id + 1]))
        bin_labels.append(val_to_put)
    
    return bins, bin_labels



bins, bin_labels = make_bins(df.Amount, size=10)
bin_labels



df['Amount-Bins'] = pd.cut(df.Amount, bins=bins,
                           labels=bin_labels, include_lowest=True)
#df['Amount-Bins'].head().to_frame()
df['Amount-Bins']



df['Amount-Bins'].value_counts()



plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df)



plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df[~(df['Amount-Bins'] == '0 to 2854')])
plt.xticks(rotation=45)



df_encoded = pd.get_dummies(data=df, columns=['Amount-Bins'])
df = df_encoded.copy()



X = df.drop(labels='Class', axis=1)
Y = df['Class']
X.shape, Y.shape



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, random_state=0, test_size=0.3, shuffle=True)

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)



from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
# Training the algorithm
lr_model.fit(xtrain, ytrain)
# Predictions on training and testing data
lr_pred_train = lr_model.predict(xtrain)
lr_pred_test = lr_model.predict(xtest)
dty = pd.DataFrame(lr_pred_test)
dty.head()




## Décision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(xtrain,ytrain)
y_pred = classifier.predict(xtest)



# Importing the required metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix



## Confusion Matrix pour Logistic Regression
tn, fp, fn, tp = confusion_matrix(ytest, lr_pred_test).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix



## Confusion Matrix pour Decision Tree
#Decision tree
tnD, fpD, fnD, tpD = confusion_matrix(ytest, y_pred).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tpD, fpD],
        'Predicted Not Fraud': [fnD, tnD]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix



## Accuracy pour Logistic Regression
lr_accuracy = accuracy_score(ytest, lr_pred_test)
lr_accuracy



## Accuracy pour l'arbre de decision 
accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred)
accuracy



from sklearn.metrics import f1_score
lr_f1 = f1_score(ytest, lr_pred_test)
lr_f1



## f1_score pour l'arbre de decision
from sklearn.metrics import f1_score
f1_new = f1_score(ytest, y_pred)
f1_new



## Classification Report pour Logistic Regression
from sklearn.metrics import classification_report
print(classification_report(ytest, lr_pred_test))



## Classification Report pour l'arbre de decision
#Decision Tree
from sklearn.metrics import classification_report
print(classification_report(ytest, y_pred))



nonfraud_indexies = df[df.Class == 0].index
fraud_indices = np.array(df[df['Class'] == 1].index)
class_val = df['Class'].value_counts()
non_fraud = class_val[0]
fraud = class_val[1]
# take random samples from non fraudulent that are equal to fraudulent samples
random_normal_indexies = np.random.choice(nonfraud_indexies, fraud, replace=False)
random_normal_indexies = np.array(random_normal_indexies)
# concatenate both indices of fraud and non fraud
under_sample_indices = np.concatenate([fraud_indices, random_normal_indexies])
#extract all features from whole data for under sample indices only
under_sample_data = df.iloc[under_sample_indices, :]

# now we have to divide under sampling data to all features & target
x_undersample_data = under_sample_data.drop(['Class'], axis=1)
y_undersample_data = under_sample_data[['Class']]
# now split dataset to train and test datasets as before
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
x_undersample_data, y_undersample_data, test_size=0.2, random_state=0)
d = pd.DataFrame(y_undersample_data)
d.value_counts()



## Amélioration du modèle Logistic Regression en utilisant Notion échantillonnage
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
# Training the algorithm
lr_model.fit(X_train_sample, y_train_sample)
# Predictions on training and testing data
pred_train = lr_model.predict(X_train_sample)
pred_test = lr_model.predict(X_test_sample)



## Amélioration du modèle arbre de décision en utilisant Notion échantillonnage
classifier_New = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_New.fit(X_train_sample,y_train_sample)
y_pred_New = classifier.predict(X_test_sample)



from sklearn.metrics import classification_report
print(classification_report(y_test_sample, pred_test))



#Decision Tree
print(classification_report(y_test_sample,y_pred_New))

