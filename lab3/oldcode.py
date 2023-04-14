from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from os import listdir
from os.path import isfile, join

def read_mails(ham_dir, spam_dir):
    ham_names = [f for f in listdir(ham_dir) if isfile(join(ham_dir, f))]
    spam_names = [f for f in listdir(spam_dir) if isfile(join(spam_dir, f))]
    ham_content = []
    spam_content = []
    for file_name in ham_names:
        with open(ham_dir+file_name, 'r', errors='ignore') as f:
            thismail = []
            thismail = f.read().replace('\n', '')
            ham_content += [thismail]
    for file_name in spam_names:
        with open(spam_dir+file_name, 'r', errors='ignore') as f:
            thismail = []
            thismail = f.read().replace('\n', '')
            spam_content += [thismail]
    
    labels = [0]*len(ham_content) + [1]*len(spam_content)
    return ham_content, spam_content, labels

ham_dir, spam_dir = 'easy_ham/', 'spam/'
ham_content, spam_content, labels = read_mails(ham_dir, spam_dir)
X_train, X_test, y_train, y_test = train_test_split(ham_content+spam_content, \
                                    labels, test_size=0.25, random_state=1337)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def disp_result(y_test, y_predict):
    print(accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict))
    mat = confusion_matrix(y_test, y_predict)
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True, \
                cmap='coolwarm', linewidths=5)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.show()

v = CountVectorizer()
X_train_count = v.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_count, y_train)
y_predict = model.predict(v.transform(X_test))
disp_result(y_test, y_predict)