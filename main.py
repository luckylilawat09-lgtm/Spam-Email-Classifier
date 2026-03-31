import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset (you can replace with a CSV later)
data = {
    'text': [
        'Win money now!!!',
        'Hey, how are you?',
        'Claim your free prize',
        'Let’s meet tomorrow',
        'Congratulations, you won a lottery!',
        'Call me when you can',
        'Exclusive offer just for you',
        'Are you coming to class?'
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham'
    ]
}

df = pd.DataFrame(data)

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.25, random_state=42
)

# Convert text to numerical data
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Predict
y_pred = model.predict(X_test_vectors)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 🔍 Test with custom input
while True:
    user_input = input("\nEnter an email message (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    
    if prediction[0] == 1:
        print("📛 This is SPAM!")
    else:
        print("✅ This is NOT spam (Ham)")