"""
Problem:

A machine learning model is requested to be developed that can predict whether individuals are diabetic or not when
their features are specified. Before developing the model, it is expected that necessary data analysis and feature
engineering steps to be performed.


Dataset Story:

The dataset is a part of a large dataset held at the National Institute of Diabetes and Digestive and Kidney Diseases
in the United States. The data is used for diabetes research conducted on Pima Indian women aged 21 and over living in
Phoenix, the fifth-largest city in the state of Arizona in the USA. The target variable is defined as "outcome",
indicating whether the diabetes test result is positive (1) or negative (0).

Pregnancies: Number of pregnancies before
Glucose: 2-hour plasma glucose concentration in the oral glucose tolerance test.
Blood Pressure: (mm Hg)
SkinThickness
Insulin: 2-hour serum insulin. (mu U/ml)
DiabetesPedigreeFunction: Function (2-hour plasma glucose concentration in the oral glucose tolerance test)
BMI: Body mass index
Age
Outcome: Having the disease (1) or not (0).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

"""
TASK 1: EXPLORATORY DATA ANALYSIS (EDA):

Step 1: Examine the overall picture
"""
df = pd.read_csv("feature_engineering/feature_engineering/CASE STUDY/diabetes/diabetes.csv")


def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("############# Number of Unique Values#################")
    for col in dataframe.columns:
        print(col + ":" + str(dataframe[col].nunique()))


check_df(df)

"""
Step 2: Capture numerical and categorical variables.
"""

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Provides the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Numerical-looking categorical variables are also included in categorical variables.

    Parameters
    ------
    dataframe: dataframe
            Dataframe whose variable names are to be obtained
    cat_th: int, optional
            class threshold value for numerical but categorical variables
    car_th: int, optional
            class threshold value for categorical but cardinal variables

    Returns
    ------
    cat_cols: list
            List of categorical variables
    num_cols: list
            List of numerical variables
    cat_but_car: list
            List of categorical but cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is within cat_cols.
        The sum of the 3 lists returned is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables
     """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
"""
Step 3: Analyze numerical and categorical variables.
"""
# categorical variable: ["Outcome"]


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)

# numerical variables: ['Pregnancies','Glucose','BloodPressure','SkinThickness',
#                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

"""
Step 4: Perform target variable analysis. 
(Mean of target variable according to categorical variables, 
mean of numerical variables according to target variable)
"""


def target_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_cols)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_with_cat(df, "Outcome", col)
# only categorical variable is target variable


def target_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_with_num(df, "Outcome", col)
"""
Step 5: Perform outlier analysis.
"""


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "BloodPressure")

"""
Step 6: Perform missing observation analysis.
"""

# Is there any missing value?
df.isnull().values.any()

# Number of missing values in variables
df.isnull().sum()

# Number of filled values in variables
df.notnull().sum()

# Total number of missing values in dataset
df.isnull().sum().sum()

# Observation units with at least one missing value
df[df.isnull().any(axis=1)]

# Complete observation units
df[df.notnull().all(axis=1)]

# Sort in descending order
df.isnull().sum().sort_values(ascending=False)

# Ratio of missing values
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# columns that have missing values
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

# no null values since they are disguised as 0.


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
missing_values_table(df, True)

msno.heatmap(df)
plt.show(block=True)


"""
Step 7: Perform correlation analysis.
"""
corr_matrix = df.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


diabetic = df[df.Outcome == 1]
healthy = df[df.Outcome == 0]

plt.scatter(healthy.Age, healthy.Insulin, color="green", label="Healthy", alpha = 0.4)
plt.scatter(diabetic.Age, diabetic.Insulin, color="red", label="Diabetic", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Insulin")
plt.legend()
plt.show(block=True)

"""
TASK 2: FEATURE ENGINEERING

Step 1: Perform the necessary operations for missing and outlier values. 
Notice that there are no missing observations in the dataset, but the observation units that contain a value of 0 in
variables such as Glucose, Insulin, etc. may represent missing values. For example, a person's glucose or insulin value
cannot be 0. Taking this into consideration, we can assign zero values as NaN in the relevant variables and then apply 
operations to missing values.
"""

df.head()

# MISSING VALUE


def potential_missing(dataframe, col_name):
    observations = dataframe[dataframe[col_name] == 0].shape[0]
    return observations


for col in num_cols:
    print(col, potential_missing(df, col))
"""
Pregnancies 111
Glucose 5
BloodPressure 35
SkinThickness 227
Insulin 374
BMI 11
DiabetesPedigreeFunction 0
Age 0
"""

na_cols = [col for col in num_cols if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

for col in na_cols:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

"""
or 
for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
"""

for col in num_cols:
    print(col, potential_missing(df, col))

df.isnull().sum()
"""
Pregnancies 111
Glucose 0
BloodPressure 0
SkinThickness 0
Insulin 0
BMI 0
DiabetesPedigreeFunction 0
Age 0
"""
# Let's check following part again:


missing_values_table(df)
missing_values_table(df, True)

msno.heatmap(df)
plt.show(block=True)


# Filling the missing values
for col in na_cols:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()
missing_values_table(df)

# OUTLIERS


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#surpressing outliers:
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

"""
Step 2: Create new variables.
"""
# from Age variable
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"
df.head()

# from BMI variable
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])
df.head()

# from Glucose variable
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])
df.head()

# from Age and BMI variables
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# from Age and Glucose variables
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

df.head()

# derive a categorical variable with insulin variable
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)
df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]
#df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * (1+ df["Pregnancies"])


df.columns = [col.upper() for col in df.columns]
df.head()

"""
Step 3: Perform encoding operations.
"""
# calling the variables again
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# label_encoder().inverse_transform()

# One-Hot Encoding
# updating cat_cols list

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
# OUTCOME is the target variable

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape
# (768, 27)

"""
Step 4: Standardize numerical variables.
"""
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape
# (768, 27)

"""
Step 5: Create a model.
"""


y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
#it measures the proportion of actual positive cases that are correctly identified as positive by the model.

print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
#it measures the proportion of positive predictions made by the model that are actually true positives.

print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
#It is the harmonic mean of precision and recall, where precision measures the proportion of true positive predictions
# out of all positive predictions, and recall measures the proportion of true positive predictions out of all actual
# positive cases.

print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
#  it measures the area under the Receiver Operating Characteristic (ROC) curve, which is a plot of the true positive
#  rate (sensitivity) against the false positive rate (1 - specificity) for different classification thresholds.


# Accuracy: 0.78
# Recall: 0.701
# Precision: 0.67
# F1: 0.68
# Auc: 0.76

##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)





