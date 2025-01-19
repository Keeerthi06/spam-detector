# spam-detector

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/content/spam.csv')

df.columns
df.info()
df.isna().sum()

df['Spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
df.head(6)

from sklearn.model_selection import train_test_split

# Assuming your DataFrame is named 'data' and it contains columns 'Category' and 'Message'
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.25, random_state=42)

# Display the shape of the splits to verify
print(f'Training data shape: {X_train.shape}')
print(f'Testing data shape: {X_test.shape}')


#CounterVectorizer Convert the text into matrics
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

clf.fit(X_train,y_train)

# List of test emails
emails = [
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e.g. HORO ARIES'
]

# Use the trained classifier to predict if the emails are spam or ham
predictions = clf.predict(emails)

# Map the numerical predictions to labels
predicted_labels = ['ham' if prediction == 0 else 'spam' for prediction in predictions]

# Display the emails with their predictions
for email, label in zip(emails, predicted_labels):
    print(f'Email: {email}\nPrediction: {label}\n')


clf.predict(emails)
clf.score(X_test,y_test)
