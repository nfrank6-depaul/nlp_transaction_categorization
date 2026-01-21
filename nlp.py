import dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# pre-process the dataset
transactions_df = dataset.preprocess()
print()
print("Data pre-processing complete.")

# set up train-test split
X = transactions_df["Description"] # only need the descriptions
y = transactions_df["Category"] # target variable is the category

# split the data and stratify to ensure the target categories are proportionally represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# initialize vectorizer for data using TF-IDF (term frequency x inverse document frequency)
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

# initialize the logistic regression classifier
clf = LogisticRegression(max_iter=2000)

# vectorize the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# train the classifier
clf.fit(X_train_tfidf, y_train)
print()
print("Logistic Regression training complete.")

# vectorize the test data
X_test_tfidf = vectorizer.transform(X_test)

# make predictions on the test data
pred = clf.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))






