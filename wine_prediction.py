import streamlit as st
import pandas as pd
import numpy as np
import Pickle
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Setting the screen size to maximum
# st.set_page_config(layout="wide")
# # col1 is the left hand sidebar
# col1 = st.sidebar
# col2, col3 = st.beta_columns((2,1)) # 2 means the middle column is bigger than the right column

st.write("""
# Wine Data Set
""")
@st.cache
def load_data():

    my_wine = load_wine()
    x=my_wine.data
    y=my_wine.target
    wine_data = np.c_[x,y]
    wine_columns = ['alcohol',
     'malic_acid',
     'ash',
     'alcalinity_of_ash',
     'magnesium',
     'total_phenols',
     'flavanoids',
     'nonflavanoid_phenols',
     'proanthocyanins',
     'color_intensity',
     'hue',
     'od280/od315_of_diluted_wines',
     'proline']
    header=wine_columns+['wine_class'] #adding a label/column called "wine_class"
    #converting into a dataframe for visualisation purpose
    wine_df = pd.DataFrame(data=wine_data, columns=header)
    return wine_df


df = load_data()


st.write(df)

st.write("""
# My predictions
""")

st.write("""
## Random Forest Predictions
""")

test_size_bar = st.sidebar.slider("Select your test size", 0.1, 0.4)
max_depth_bar = st.sidebar.slider("Select your max depth", 2, 10)

X = df.drop(['wine_class'], axis=1)
y = df['wine_class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size_bar,random_state=101)
clf = RandomForestClassifier(max_depth=max_depth_bar, random_state=101)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_preds, rownames=['Actual'], colnames = ['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
accur = accuracy_score(y_test, y_preds)

st.write("""
### The accuracy score is calculated by sum(allcorrect)/sum(correct+incorrect) = accuracy_score""")
st.write("""
# Accuracy Score is """)
st.write({accur})

st.write("""
### The below is the confusion matrix use to calculate the accuracy score""")

st.pyplot()



if st.button("Show feature importance"):

    st.write("""
    ## Let's check for important features""")

    feature_importance_df = pd.DataFrame({"feature": list(X.columns), "importance": clf.feature_importances_}
                                         ).sort_values("importance", ascending=False)

    st.write(feature_importance_df)

    # Creating a bar plot
    sns.barplot(x=feature_importance_df.feature, y=feature_importance_df.importance)
    # Add labels to your

    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Visualizing Important Features")
    plt.xticks(
        rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large"
    )
    st.pyplot()

    st.write("""
    ### Now that we know which ones are least important, we can drop those columns and re-run our prediction.""")

    X = df.drop(['wine_class', 'magnesium', 'ash'], axis=1)
    y = df['wine_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    clf = RandomForestClassifier(max_depth=2, random_state=101)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    confusion_matrix = pd.crosstab(y_test, y_preds, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, )
    accur = accuracy_score(y_test, y_preds)

    st.subheader(f"""
    Accuracy Score of# {accur} is the same even after removing those columns """)
