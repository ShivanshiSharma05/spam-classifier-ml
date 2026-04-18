# ===== DAY 1: DATA EXPLORATION =====

#import libraries
import pandas as pd

#load dataset
df=pd.read_csv("spam.csv", encoding="latin-1")

#print first five rows
print("\n  ")
print("DATA EXPLORATION")
print("\nfirst five rows:")
print(df.head())

#print coulumn names
print("\ncolumn names:")
print(df.columns)

#print dataset shape
print("\ndataset shape:")
print(df.shape)

#count Spams and Hams
print("\nSpams and Hams count:")
print(df['v1'].value_counts())

#-------------------------------------------------------------------------------------------

# ===== DAY 2: TEXT CLEANING =====

#step1: keep only required col(remove others)
df=df[['v1','v2']]

#step2: rename columns
df.rename(columns={'v1':'label','v2':'message'},inplace=True)

#step3: text cleaning function
import re
def clean_text(text):
    text=text.lower() #lowercase
    text=re.sub(r'[^a-z\s]',' ',text) # remove punctuation & numbers
    text=re.sub(r'\s+',' ',text).strip() # remove extra space
    return text

#step4: apply cleaning
df['message']=df['message'].apply(clean_text)

# CHECK : print first 2 rows
for i in range(2):
    print("\n  ")
    print("\nTEXT CLEANING")

    print("\noriginal:", df['message'].iloc[i])

    print("\ncleaned:",df['message'].iloc[i])


#-------------------------------------------------------------------------------------------

# ===== DAY 3: Feature Extraction ( text -> numbers) =====

# step1 : seperate features and labels
X=df['message']
Y=df['label']

# step2 : Encode Labels
Y=Y.map({'ham': 0, 'spam' : 1})

# step3 : Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_vectorized=cv.fit_transform(X)

# step4 : Inspect Output
print("\n  ")
print("\nFeature Extraction")
print("\nshape of vectorized X:",X_vectorized.shape)
print("\nfirst 10 words from vocabulary:",cv.get_feature_names_out()[:10])


#-------------------------------------------------------------------------------------------

# ===== DAY 4: Model Training =====

# step1 : train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_vectorized,Y,test_size=0.2,random_state=42)

# step 2 & 3 : create and train the model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,Y_train)

# step4 : make predictions
y_pred=model.predict(X_test)

# step5 : quick sanity check
print("\n  ")
print("\nModel Training")
print("\nfirst 10 predictions:",y_pred[:10])
print("\nfirst 10 actual labels:",Y_test.values[:10])


#-------------------------------------------------------------------------------------------

# ===== DAY 5 : Model Evaluation =====

# step1 : accuracy
print("\n  ")
print("\nModel Evaluation")

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,y_pred)
print("\nAccuracy Score:",accuracy)

# step2 : confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
print("\nConfusion Matrix:",cm)

# step3 : classification report
from sklearn.metrics import classification_report
cr=classification_report(Y_test,y_pred,target_names=['Ham','Spam'])
print("\nClassification Report:",cr)


#-------------------------------------------------------------------------------------------

# ===== DAY 6 : MODEL IMPROVEMENT (TF-IDF + COMPARISON)=====

from sklearn.feature_extraction.text import TfidfVectorizer

#step1 : vectorize text again
tfidf=TfidfVectorizer()
X_tfidf=tfidf.fit_transform(X)
Y_tfidf=Y

#step2 : train-test split
X_train,X_test,Y_train,Y_test=train_test_split(X_tfidf,Y_tfidf,test_size=0.2,random_state=42)

#step3 : train naive model again
nb_tfidf_model=MultinomialNB()
nb_tfidf_model.fit(X_train,Y_train)

#step4 : evaluate again
y_pred_tfidf=nb_tfidf_model.predict(X_test)

print("\n  ")
print("\nModel Improvement")
print("\ntfidf accuracy score:", accuracy_score(Y_test, y_pred_tfidf))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_pred_tfidf))
print("\nClassification Report:\n", classification_report(Y_test, y_pred_tfidf))

# --- Step 6: Compare ---
"""
Comparison Analysis:
TF-IDF usually performs slightly better (or similarly) compared to CountVectorizer because it penalizes 
frequently occurring words (like 'the', 'is') that carry less information, while giving more weight to 
unique, discriminative words. In spam detection, this helps the model ignore common language noise 
and focus on specific keywords that actually indicate spam.
"""

#-------------------------------------------------------------------------------------------
# ===== DAY 7 : REAL-TIME SPAM DETECTION =====

def predict_message(message, use_tfidf=True):
    message = clean_text(message)
    
    if use_tfidf:
        vector = tfidf.transform([message])
        prediction = nb_tfidf_model.predict(vector)
    else:
        vector = cv.transform([message])
        prediction = model.predict(vector)

    return "🚨 SPAM" if prediction[0] == 1 else "✅ HAM (Not Spam)"


print("\n")
print("DAY 7 : REAL MESSAGE TESTING")

msg1 = "Congratulations! You have won a free lottery ticket. Call now!"
msg2 = "Hi, are you coming to class tomorrow?"

print("\nMessage:", msg1)
print("Prediction:", predict_message(msg1))

print("\nMessage:", msg2)
print("Prediction:", predict_message(msg2))
