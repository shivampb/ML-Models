import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import streamlit as st

# Load and preprocess the data
data = pd.read_csv('spam.csv')
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Input and output separation
message = data['Message']
cat = data['Category']

# Convert text to numerical format using CountVectorizer
vectorizer = CountVectorizer()
message_transformed = vectorizer.fit_transform(message)

# Splitting the data for training and testing
mess_train, mess_test, cat_train, cat_test = train_test_split(
    message_transformed, cat, train_size=0.8, random_state=42
)

# Creating the Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(mess_train, cat_train)

# Making predictions
result = model.predict(mess_test)

# Evaluating the model
accuracy = accuracy_score(cat_test, result)
# print(f"Accuracy: {accuracy * 100:.2f}%")


#Predicting Model
def predict(message:str)->str:
    input_message = vectorizer.transform([message]).toarray()
    result = model.predict(input_message)
    return result

st.header("Email Spam Detector")
input_message = st.text_input("Enter Email To Check")

if st.button("CHECK"):
    if not input_message.strip():  # Check for empty or whitespace input
        st.write("Please enter a message to check.")
    else:
        res = predict(input_message)    
        if res == 'Spam':
            st.write("It's Spam Bro")
        elif res == 'Not Spam':
            st.write("It's Not a Spam Bro")
        else:
            st.write("Invalid input or model error. Please try again.")