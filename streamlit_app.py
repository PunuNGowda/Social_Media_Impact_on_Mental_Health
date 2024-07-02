import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Function to clean tweets
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\S+)", " ", tweet).split())

# Function to get the polarity of the tweet
def textbl(tweet):
    text = clean_tweet(tweet)
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to classify tweets
def sent(tweet):
    text = clean_tweet(tweet)
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.5:
        return 0
    else:
        return 1

# Streamlit app
st.title("Depression Detection in Tweets")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# File uploader
uploaded_file = st.file_uploader("Choose a JSON file", type="json")

# Submit button
if st.button('Submit'):
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            # Load the dataset
            tweets = json.load(uploaded_file)
            list_tweets = [list(elem.values()) for elem in tweets]
            list_columns = list(tweets[0].keys())
            df = pd.DataFrame(list_tweets, columns=list_columns)
            
            # Process the dataset
            df['Depressed'] = np.array([str(sent(tweet)) for tweet in df['text']])
            d = df.drop(['user', 'text', 'url', 'fullname', 'timestamp', 'id', 'html'], axis=1)
            
            y = d['Depressed']
            X = d.drop('Depressed', axis=1)
            tot_count = len(df.index)
            st.write(f"Total number of tweets: {tot_count}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
            st.session_state.df = df
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # KNN Algorithm
            knn = KNeighborsClassifier(n_neighbors=3)
            trained_knn = knn.fit(X_train, y_train)
            acc_knn = round(knn.score(X_train, y_train) * 100, 2)
            acc_test_knn = round(knn.score(X_test, y_test) * 100, 2)
            y_pred_knn = knn.predict(X_test)
            
            # Logistic Regression
            logistic_regression_model = LogisticRegression()
            trained_logistic_regression_model = logistic_regression_model.fit(X_train, y_train)
            acc_train_lr = round(trained_logistic_regression_model.score(X_train, y_train) * 100, 2)
            acc_test_lr = round(trained_logistic_regression_model.score(X_test, y_test) * 100, 2)
            y_pred_lr = logistic_regression_model.predict(X_test)
            
            # Random Forest
            random_forest_model = RandomForestClassifier(n_estimators=600)
            trained_random_forest_model = random_forest_model.fit(X_train, y_train)
            acc_train_rf = round(trained_random_forest_model.score(X_train, y_train) * 100, 2)
            acc_test_rf = round(trained_random_forest_model.score(X_test, y_test) * 100, 2)
            y_pred_rf = random_forest_model.predict(X_test)
            
            # Store results in session state
            st.session_state.results = {
                'KNN': {
                    'train_accuracy': acc_knn,
                    'test_accuracy': acc_test_knn,
                    'predictions': y_pred_knn,
                    'probs': knn.predict_proba(X_test)[:, 1]
                },
                'Logistic Regression': {
                    'train_accuracy': acc_train_lr,
                    'test_accuracy': acc_test_lr,
                    'predictions': y_pred_lr,
                    'probs': logistic_regression_model.predict_proba(X_test)[:, 1]
                },
                'Random Forest': {
                    'train_accuracy': acc_train_rf,
                    'test_accuracy': acc_test_rf,
                    'predictions': y_pred_rf,
                    'probs': random_forest_model.predict_proba(X_test)[:, 1]
                }
            }
            
            # Display results
            st.write("### Model Accuracy")
            st.write(f"KNN Training Accuracy: {acc_knn}%")
            st.write(f"KNN Testing Accuracy: {acc_test_knn}%")
            st.write(f"Logistic Regression Training Accuracy: {acc_train_lr}%")
            st.write(f"Logistic Regression Testing Accuracy: {acc_test_lr}%")
            st.write(f"Random Forest Training Accuracy: {acc_train_rf}%")
            st.write(f"Random Forest Testing Accuracy: {acc_test_rf}%")
            
            # Determine the most accurate algorithm
            accuracy_dict = {
                'KNN': acc_test_knn,
                'Logistic Regression': acc_test_lr,
                'Random Forest': acc_test_rf
            }
            most_accurate_algorithm = max(accuracy_dict, key=accuracy_dict.get)
            st.write(f"### Most Accurate Algorithm: {most_accurate_algorithm} ({accuracy_dict[most_accurate_algorithm]}%)")
            
# Dropdown box for algorithm selection
if st.session_state.results:
    st.write("### Select an Algorithm")
    algorithm = st.selectbox(
        'Choose an algorithm',
        ('KNN', 'Logistic Regression', 'Random Forest')
    )
    
    # Results based on selected algorithm
    if algorithm:
        probs = st.session_state.results[algorithm]['probs']
        y_pred = st.session_state.results[algorithm]['predictions']
        
        # Display classification report and confusion matrix
        st.write("### Classification Report")
        st.text(classification_report(st.session_state.y_test, y_pred))
        st.write("### Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        # Separate datasets
        not_depressed_df = st.session_state.df[st.session_state.df['Depressed'] == '1']
        depressed_df = st.session_state.df[st.session_state.df['Depressed'] == '0']
        
        # Download links for the datasets
        not_depressed_csv = not_depressed_df.to_csv(index=False).encode('utf-8')
        depressed_csv = depressed_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Not Depressed Dataset",
            data=not_depressed_csv,
            file_name='not_depressed.csv',
            mime='text/csv',
        )
        
        st.download_button(
            label="Download Depressed Dataset",
            data=depressed_csv,
            file_name='depressed.csv',
            mime='text/csv',
        )
        
        # Plotting options
        st.write("### Select Graphs to Display")
        show_counts = st.checkbox('Show Counts of Depressed and Not Depressed')
        show_percentages = st.checkbox('Show Percentage of Depressed and Not Depressed')
        show_comparison = st.checkbox('Show Comparison of Model Accuracies')
        show_precision_recall = st.checkbox('Show Precision and Recall of Algorithms')
        show_roc = st.checkbox('Show ROC Curves of Algorithms')
        
        # Plotting results
        if show_counts:
            st.write("### Counts of Depressed and Not Depressed")
            depressed_counts = st.session_state.df['Depressed'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=depressed_counts.index, y=depressed_counts.values, palette="Blues_d", ax=ax)
            ax.set_xlabel('Depression Status')
            ax.set_ylabel('Count')
            ax.set_title('Counts of Depressed and Not Depressed')
            st.pyplot(fig)
        
        if show_percentages:
            st.write("### Percentage of Depressed and Not Depressed")
            depressed_percentages = st.session_state.df['Depressed'].value_counts(normalize=True) * 100
            fig, ax = plt.subplots()
            labels = ['Not Depressed', 'Depressed']
            colors = ['#ff9999','#66b3ff']
            ax.pie(depressed_percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title('Percentage of Depressed and Not Depressed')
            st.pyplot(fig)
        
        if show_comparison:
            st.write("### Comparison of Model Accuracies")
            algorithms = ['KNN', 'Logistic Regression', 'Random Forest']
            train_accuracies = [
                st.session_state.results['KNN']['train_accuracy'],
                st.session_state.results['Logistic Regression']['train_accuracy'],
                st.session_state.results['Random Forest']['train_accuracy']
            ]
            test_accuracies = [
                st.session_state.results['KNN']['test_accuracy'],
                st.session_state.results['Logistic Regression']['test_accuracy'],
                st.session_state.results['Random Forest']['test_accuracy']
            ]
            
            fig, ax = plt.subplots()
            bar_width = 0.35
            index = np.arange(len(algorithms))
            
            rects1 = ax.bar(index, train_accuracies, bar_width, label='Training Accuracy', color='b')
            rects2 = ax.bar(index + bar_width, test_accuracies, bar_width, label='Testing Accuracy', color='g')
            
            ax.set_xlabel('Algorithms')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training vs Testing Accuracy of Algorithms')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(algorithms)
            ax.legend()
            st.pyplot(fig)
        
        if show_precision_recall:
            st.write("### Precision and Recall of Algorithms")
            precision_knn, recall_knn, _ = precision_recall_curve(st.session_state.y_test.astype(float), st.session_state.results['KNN']['probs'])
            precision_lr, recall_lr, _ = precision_recall_curve(st.session_state.y_test.astype(float), st.session_state.results['Logistic Regression']['probs'])
            precision_rf, recall_rf, _ = precision_recall_curve(st.session_state.y_test.astype(float), st.session_state.results['Random Forest']['probs'])
            
            fig, ax = plt.subplots()
            ax.plot(recall_knn, precision_knn, label='KNN', color='r')
            ax.plot(recall_lr, precision_lr, label='Logistic Regression', color='b')
            ax.plot(recall_rf, precision_rf, label='Random Forest', color='g')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision and Recall of Algorithms')
            ax.legend()
            st.pyplot(fig)
        
        if show_roc:
            st.write("### ROC Curves of Algorithms")
            fpr_knn, tpr_knn, _ = roc_curve(st.session_state.y_test.astype(float), st.session_state.results['KNN']['probs'])
            fpr_lr, tpr_lr, _ = roc_curve(st.session_state.y_test.astype(float), st.session_state.results['Logistic Regression']['probs'])
            fpr_rf, tpr_rf, _ = roc_curve(st.session_state.y_test.astype(float), st.session_state.results['Random Forest']['probs'])
            
            fig, ax = plt.subplots()
            ax.plot(fpr_knn, tpr_knn, label='KNN', color='r')
            ax.plot(fpr_lr, tpr_lr, label='Logistic Regression', color='b')
            ax.plot(fpr_rf, tpr_rf, label='Random Forest', color='g')
            ax.plot([0, 1], [0, 1], linestyle='--', color='k')
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)
else:
    st.error("Please upload a JSON file and click 'Submit'.")
