#loading packages:
import pandas as pd
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix, classification_report, accuracy_score, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedStratifiedKFold,StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


#loading dataset
df = pd.read_csv('/Users/----/Desktop/Stroke_prediction/healthcare-dataset-stroke-data.csv')
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()
df.bmi.replace(to_replace=np.nan, value= df.bmi.mean(), inplace=True)

#Percentage of patients who have a stroke
perc_dis = df['stroke'].sum() / len(df)
print('Percent of patients in dataset with stroke:', round(perc_dis, 4))

#Vizualizing corelation
sns.heatmap(df.corr(), annot=True, fmt='.2f', linewidths=.5,cmap="YlGnBu")

#Dropping ID column
df.drop(['id'], axis=1, inplace=True)

#get all object features
obj_feat = df.dtypes[df.dtypes == 'O'].index.values
le = LabelEncoder()
for i in obj_feat:
    df[i] = le.fit_transform(df[i])
df.head()

#Preparing variables for prediction
X = df.drop('stroke', axis=1)
y = df['stroke']

#Scaling and normalizing predictors
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
X.head()
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_minmax, y,test_size=0.2,random_state=42, stratify = y)

#### --- INITIAL TEST --- ####
#Define model to be tested 
all_model = [LogisticRegression(), KNeighborsClassifier(), SVC(), DecisionTreeClassifier(),
            RandomForestClassifier(),  GradientBoostingClassifier()]

#Quality measure
recall = []
accuracy = []
precision = []

for model in all_model:
    cv = cross_val_score(model, X_train, y_train, scoring='recall', cv=10).mean()
    recall.append(cv)

    cv = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10).mean()
    accuracy.append(cv)

    cv = cross_val_score(model, X_train, y_train, scoring='precision', cv=10).mean()
    precision.append(cv)

model = ['LogisticRegression', 'KNeighborsClassifier', 'SVC', 'DecisionTreeClassifier',
         'RandomForestClassifier', 'GradientBoostingClassifier']

score = pd.DataFrame({'Model': model, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall})
score.style.background_gradient(high=1,axis=0)

#Confusion matrix 
#Logistic regression 
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
y_pred = log_model.predict(X_test)
y_pred_proba = log_model.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Balancing the data for stroke and non-stroke
from imblearn.over_sampling import SMOTENC #balance classes
smote = SMOTENC([1,2,3,4,7,8,9])  #index of input numerical features
X_train , y_train = smote.fit_resample(X_train, y_train)
features = ['age','hypertension','heart_disease','ever_married','Residence_type','avg_glucose_level','bmi','gender','work_type','smoking_status']
X_train = pd.DataFrame(data = X_train, columns = features)
X_test = pd.DataFrame(data = X_test, columns = features)

#### --- LOGISTIC REGRESSION --- ####
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X_test)
y_pred_proba = log.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
scores = cross_validate(log, X_train, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
log.get_params().keys()


#Choose hyperparameter values 
log_params={'class_weight':[None,'balanced'], 
        'C': [1,2,3,4,5,6,7,8,9,10],  
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
         'intercept_scaling': [0.1,0.5,0.8,1,1.2,1.5,1.8,2], 
         'max_iter': [50,100,150,200]}

#Run the random search
rslog = RandomizedSearchCV(log_model,log_params,#model and parameters
                             cv=10,#number of cross validation folds
                             scoring='recall',#accuracy metric
                             n_iter=1)#number of random parameter combinations
rslog.fit(X_train,y_train)

#Look at the parameters for the best model
rslog.best_estimator_

# Visualizing RF results as a confusion matrix
rslog.best_estimator_.fit(X_train, y_train)
plot_confusion_matrix(clflog.best_estimator_,X_test,y_test,cmap=plt.cm.Blues,values_format = '.5g', display_labels = ["No Stroke","Stroke"])
plt.show()

#Vizualize AUC
Roc = RocCurveDisplay.from_estimator(rslog, X_test, y_test)
_ = Roc.ax_.set_title("AUC")

#Final optimized model
log_op =LogisticRegression(C=9, class_weight='balanced', dual=None,
                   intercept_scaling=0.8, max_iter=200, solver='saga')
log_op.fit(X_train,y_train)


###### --- DECISION TREE CLASSIFIER --- #####
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
dtc.get_params(['n_estimators'])
dtc.get_params().keys()

#Choose some hyperparameter values 
dtc_params={'max_features':[None,'auto','sqrt'],
                  'class_weight':[None,'balanced'],
                  'criterion': ['gini','entropy'],
                  'max_depth': [None,10, 50, 100, 250, 300 ,400],
                  'min_samples_split': [2,3,4,5,6,10,20]}

#Run the random search
dtc_rs = RandomizedSearchCV(dtc,dtc_params,#model and parameters
                             cv=10,#number of cross validation folds
                             scoring='roc_auc',#accuracy metric
                             n_iter=1)#number of random parameter combinations
dtc_rs.fit(X_train,y_train)

#Look at the parameters for the best model
dtc_rs.best_estimator_

# Visualizing RF results as a confusion matrix
dtc_rs.best_estimator_.fit(X_train, y_train)
plot_confusion_matrix(dtc_rs.best_estimator_,X_test,y_test,cmap=plt.cm.Blues,values_format = '.5g', display_labels = ["No Stroke","Stroke"])
plt.show()

#vizualize AUC
ROC_dtc = RocCurveDisplay.from_estimator(dtc_rs, X_test, y_test)
_ = ROC_dtc.ax_.set_title("AUC")

#Final optimized model
dtc_op = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features='sqrt',
                       min_samples_split=10)
dtc_op.fit(X_train,y_train)
y_pred = dtc_op.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Checking which variable had the strongest impact on the model performance
pd.DataFrame(dtc_op.feature_importances_, index=X.columns, columns=['Feature Importance']).sort_values(by='Feature Importance').plot.bar()



##### --- KNEIGHBORS CLASSIFIER --- #####
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
knn.get_params(['n_estimators'])
knn.get_params().keys()

knn_params={'n_neighbors':[1,2,3,4,5,6,7,8,9,10,13, 15, 17, 19],
            'metric': ['euclidean', 'manhattan', 'minkowski']}

#Run the random parameter search
knn_rs = RandomizedSearchCV(knn,knn_params,#model and parameters
                             cv=10,#number of cross validation folds
                             scoring='roc_auc',#accuracy metric
                             n_iter=1)#number of random parameter combinations
knn_rs.fit(X_train,y_train)

#Look at the parameters for the best model
knn_rs.best_estimator_

# Visualizing RF results as a confusion matrix
knn_rs.best_estimator_.fit(X_train, y_train)
plot_confusion_matrix(knn_rs.best_estimator_,X_test,y_test,cmap=plt.cm.Blues,values_format = '.5g', display_labels = ["No Stroke","Stroke"])
plt.show()
y_pred = knn_rs.predict(X_test)
y_pred_proba = knn_rs.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Vizualize AUC
PreRec_knn = PrecisionRecallDisplay.from_estimator(
    knn_rs, X_test, y_test, name="DecisionTreeClassifier")
_ = PreRec_knn.ax_.set_title("2-class Precision-Recall curve")
ROC_knn = RocCurveDisplay.from_estimator(knn_rs, X_test, y_test)
_ = ROC_knn.ax_.set_title("AUC")

#Final optimized model
knn_op = KNeighborsClassifier(metric='euclidean', n_neighbors=7)
knn_op.fit(X_train,y_train)


##### --- SVM classifier --- #####
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
svc.get_params(['n_estimators'])
svc.get_params().keys()

svc_params={'C':[1,2,3,4],
                  'kernel':['precomputed','rbf', 'linear', 'poly', 'sigmoid'],
                   'degree': [3,9],
                   'random_state': [None,1,2]}

#Run the random parameter search
svc_rs = RandomizedSearchCV(svc,svc_params,#model and parameters
                             cv=10,#number of cross validation folds
                             scoring='roc_auc',#accuracy metric
                             n_iter=1)#number of random parameter combinations
svc_rs.fit(X_train,y_train)

#Look at the parameters for the best model
svc_rs.best_estimator_

# Visualizing RF results as a confusion matrix
svc_rs.best_estimator_.fit(X_train, y_train)
plot_confusion_matrix(svc_rs.best_estimator_,X_test,y_test,cmap=plt.cm.Blues,values_format = '.5g', display_labels = ["No Stroke","Stroke"])
plt.show()
y_pred = svc_rs.predict(X_test)
y_pred_proba = knn_rs.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Vizualize AUC
ROC_svc = RocCurveDisplay.from_estimator(svc_rs, X_test, y_test)
_ = ROC_svc.ax_.set_title("AUC")

#Final optimized model
svc_op = SVC(C=1, degree=9, kernel='linear', random_state=2)
svc_op.fit(X_train,y_train)



##### ENSEMBLE APPROACHES #####

##### -- RANDOM FOREST CLASSIFIER -- #####
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
y_pred_proba = rfc.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
rfc.get_params(['n_estimators'])
rfc.get_params().keys()

#Choose hyperparameter values 
rfc_params={'max_features':['auto','sqrt'],
                  'class_weight':[None,'balanced'],
                  'max_samples': [0,1,5,10,20,50,100],
                  'min_samples_split': [2,3,4,5,6,10,20],
                  'max_depth': [None,2, 5, 10, 50, 100, 250, 300 ,400],
                  'n_estimators': [100,200,400,600,800,1000],
                  'ccp_alpha':[0, 0.001, 0.005, 0.01, 0.05]}

#Run the random search
rfc_sc = RandomizedSearchCV(rfc,rfc_params,#model and parameters
                             cv=10,#number of cross validation folds
                             scoring='roc_auc',#accuracy metric
                             n_iter=1)#number of random parameter combinations
rfc_sc.fit(X_train,y_train)

#Look at the parameters for the best model
rfc_sc.best_estimator_

# Visualizing RF results as a confusion matrix
rfc_sc.best_estimator_.fit(X_train, y_train)
plot_confusion_matrix(rfc_sc.best_estimator_,X_test,y_test,cmap=plt.cm.Blues,values_format = '.5g', display_labels = ["No Stroke","Stroke"])
plt.show()
y_pred = rfc_sc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Adjusted model
rfc = RandomForestClassifier(ccp_alpha=0.001, max_depth=100, max_features='sqrt',
                       max_samples=20, min_samples_split=10)
rfc.fit(X_train, y_train)
pd.DataFrame(rfc.feature_importances_, index=X.columns, columns=['Feature Importance']).sort_values(by='Feature Importance').plot.bar()

#Vizualize AUC
ROC_rfc = RocCurveDisplay.from_estimator(rfc_sc, X_test, y_test)
_ = ROC_rfc.ax_.set_title("AUC")

#Final optimized model
rfc_op = RandomForestClassifier(ccp_alpha=0.001, max_depth=100, max_features='sqrt',
                       max_samples=20, min_samples_split=10)
rfc_op.fit(X_train,y_train)



##### --- GRADIENT BOOSTING CLASSIFIER --- ######
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
y_pred_proba = gbc.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
gbc.get_params(['n_estimators'])
gbc.get_params().keys()

gbc_params={'ccp_alpha':[1e-1, 1, 1e1],
                'learning_rate':[0.1,0.05,0.01,0.001],
                'criterion': ['friedman_mse', 'squared_error', 'mse', 'mae'],
                'max_features': ['auto', 'sqrt', 'log2'],
                  'n_estimators': [400,700,1000],
                 'random_state':[0],
                 'max_depth' : [2, 5, 10, 15, None], 
                 'min_samples_split' : [2,3,4,5,6,7,8,9,10],
                 'min_samples_leaf': [None,1,2,5,10], 
                 'max_features': ['log2', 'sqrt', 'auto', 'None'],
                 'loss':['deviance', 'exponential']}

#Run the random search
gbc_sc = RandomizedSearchCV(gbc,gbc_params,#model and parameters
                             cv=10,#number of cross validation folds
                             scoring='roc_auc',#accuracy metric
                             n_iter=1)#number of random parameter combinations
gbc_sc.fit(X_train,y_train)

#Look at the parameters for the best model
gbc_sc.best_estimator_

# Visualizing RF results as a confusion matrix
gbc_sc.best_estimator_.fit(X_train, y_train)
plot_confusion_matrix(rfc_sc.best_estimator_,X_test,y_test,cmap=plt.cm.Blues,values_format = '.5g', display_labels = ["No Stroke","Stroke"])
plt.show()
y_pred = gbc_sc.predict(X_test)
y_pred_proba = gbc_sc.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

gbc = GradientBoostingClassifier(ccp_alpha=0.1, criterion='mse', loss='exponential',
                           max_depth=5, max_features='log2',
                           min_samples_leaf=10, min_samples_split=6,
                           n_estimators=1000, random_state=0)
gbc.fit(X_train, y_train)
pd.DataFrame(gbc.feature_importances_, index=X.columns, columns=['Feature Importance']).sort_values(by='Feature Importance').plot.bar()

#Vizualize AUC
PreRec = PrecisionRecallDisplay.from_estimator(
    gbc_sc, X_test, y_test, name="DecisionTreeClassifier")
_ = PreRec.ax_.set_title("2-class Precision-Recall curve")
ROC_gbc = RocCurveDisplay.from_estimator(gbc_sc, X_test, y_test)
_ = ROC_gbc.ax_.set_title("AUC")

#Final model optimized
gbc_op = GradientBoostingClassifier(ccp_alpha=0.1, criterion='mse', loss='exponential',
                           max_depth=5, max_features='log2',
                           min_samples_leaf=10, min_samples_split=6,
                           n_estimators=1000, random_state=0)
gbc_op.fit(X_train,y_train)


####### --- Comparing models: VIZUALIZATIONS --- ######

all_model = [log_op, dtc_op, knn_op, svc_op,rfc_op, gbc_op]

#Quality measure
recall = []
accuracy = []
precision = []

for model in all_model:
    cv = cross_val_score(model, X_train, y_train, scoring='recall', cv=10).mean()
    recall.append(cv)

    cv = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10).mean()
    accuracy.append(cv)

    cv = cross_val_score(model, X_train, y_train, scoring='precision', cv=10).mean()
    precision.append(cv)

model = ['LogisticRegression', 'KNeighborsClassifier', 'SVC', 'DecisionTreeClassifier',
         'RandomForestClassifier', 'GradientBoostingClassifier']

score = pd.DataFrame({'Model': model, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall})
score.style.background_gradient(high=1,axis=0)

fig = plot_roc_curve(log_op, X_test, y_test)
fig = plot_roc_curve(dtc_op, X_test, y_test, ax = fig.ax_)
fig = plot_roc_curve(knn_op, X_test, y_test, ax = fig.ax_)
fig = plot_roc_curve(svc_op, X_test, y_test, ax = fig.ax_)
fig = plot_roc_curve(rfc_op, X_test, y_test, ax = fig.ax_)
fig = plot_roc_curve(gbc_op, X_test, y_test, ax = fig.ax_)
fig.figure_.suptitle("ROC curve comparison")
plt.show() 
             
#### --- Combining different model to improve overall prediction --- ####
from sklearn.ensemble import StackingClassifier
estimators = [('svc', SVC(C=1, degree=9, kernel='linear', random_state=2)), 
('gbc', GradientBoostingClassifier(ccp_alpha=0.1, criterion='mse', loss='exponential',
                           max_depth=5, max_features='log2',
                           min_samples_leaf=10, min_samples_split=6,
                           n_estimators=1000, random_state=0)),
    ('log', LogisticRegression(C=9, class_weight='balanced', dual= None,
                   intercept_scaling=0.8, max_iter=200, solver='saga'))]

stc = StackingClassifier(estimators=estimators)
stc.fit(X_train, y_train).score(X_test, y_test)

PreRec = PrecisionRecallDisplay.from_estimator(
    stc, X_test, y_test, name="Stacking Classifier")
_ = PreRec.ax_.set_title("Precision-Recall curve")
ROC_stc = RocCurveDisplay.from_estimator(stc, X_test, y_test)
_ = ROC_stc.ax_.set_title("AUC")

y_pred = stc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

pd.DataFrame(stc.feature_importances_, index=X.columns, columns=['Feature Importance']).sort_values(by='Feature Importance').plot.bar()

#########################################################################
#Neural Network using Keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import talos
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/-----/Desktop/healthcare-dataset-stroke-data.csv')
df.bmi.replace(to_replace=np.nan, value= df.bmi.mean(), inplace=True)

#Dropping ID column
df.drop(['id'], axis=1, inplace=True)

#get all object features
obj_feat = df.dtypes[df.dtypes == 'O'].index.values
le = LabelEncoder()
for i in obj_feat:
    df[i] = le.fit_transform(df[i])
df.head()

#Preparing variables for prediction
X = df.drop('stroke', axis=1)
y = df['stroke']

#Scaling and normalizing predictors
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_minmax, y,test_size=0.2,random_state=42, stratify = y)

#Balancing the data for stroke and non-stroke
from imblearn.over_sampling import SMOTENC #balance classes
smote = SMOTENC([1,2,3,4,7,8,9])  #index of input numerical features
X_train , y_train = smote.fit_resample(X_train, y_train)
features = ['age','hypertension','heart_disease','ever_married','Residence_type','avg_glucose_level','bmi','gender','work_type','smoking_status']
X_train = pd.DataFrame(data = X_train, columns = features)
X_test = pd.DataFrame(data = X_test, columns = features)
input_shape = [X_train.shape[1]]

#Creating model 
model = keras.Sequential([
    # input layer
    layers.BatchNormalization(input_shape=input_shape),
    # hidden layer 1
    layers.Dense(units=256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.4),
    # hidden layer 2
    layers.Dense(units=128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.4),
    # hidden layer 3
    layers.Dense(units=64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.4),
    # hidden layer 4
    layers.Dense(units=32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.4),
    layers.Dense(units=3, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=200,
    epochs=1000,
    verbose=0
)

### Loss Graph
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Learning Curve: Loss over Epochs")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['Training Loss', 'Validation Loss'])

### Accuracy Graph
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Learning Curve: Accuracy over Epochs")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['Training Accuracy', 'Validation Accuracy'])


#Confusion matrix 
label_names = ["Had no stroke", "Had stroke"] # 0 patient had no stroke, 1 patient had stroke
y_actual = y_train.to_numpy()
y_pred = model.predict(X_train, verbose=0)
y_pred = np.argmax(y_pred, axis=-1)

print("On {} samples of untrained(test) dataset:".format(len(y_pred)))
print("Prediction:")
print(y_pred)
print("Actual:")
print(y_actual)

### Classification Report
print("\nClassification Report:")
print(classification_report(y_actual,y_pred, target_names=label_names))

### Confusion Matrix Graph
cm = confusion_matrix(y_true=y_actual, y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()
plt.title('Confusion Matrix')




