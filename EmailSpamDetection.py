import pandas as pd


df = pd.read_csv('Spam email dataset.csv',encoding='Latin')


random_data=df.sample(frac=1, random_state=1)

training_test_index=round(len(random_data)*0.8)

training_set=random_data[:training_test_index].reset_index(drop=True)
test_set=random_data[training_test_index:].reset_index(drop=True)


training_set["Email Subject"] = training_set["Email Subject"].str.replace("\W"," ")
training_set["Email Subject"] = training_set["Email Subject"].str.lower()
training_set["Email Subject"] = training_set["Email Subject"].str.split()

keywords = []

for i in training_set["Email Subject"]:
    for words in i:
        keywords.append(words)


word_counts_per_mail = {unique_word: [0] * len(training_set["Email Subject"]) for unique_word in keywords}

for index, mail in enumerate(training_set["Email Subject"]):
   for word in mail:
      word_counts_per_mail[word][index] += 1
df1 = pd.DataFrame(word_counts_per_mail)


training_set_modified = pd.concat([training_set,df1],axis = 1)

Spam = training_set_modified[training_set_modified['Spam'] == 1.0]
Ham = training_set_modified[training_set_modified['Spam'] == 0.0]

percentage_Spam = len(Spam) / len(training_set_modified)
percentage_Ham = len(Ham) / len(training_set_modified)

n_words_per_spam_message = Spam['Email Subject'].apply(len)
n_spam = n_words_per_spam_message.sum()

n_words_per_ham_message = Ham['Email Subject'].apply(len)
n_ham = n_words_per_ham_message.sum()

n_keywords = len(keywords)

alpha = 1

parameters_spam = {unique_word:0 for unique_word in keywords}
parameters_ham = {unique_word:0 for unique_word in keywords}


for word in keywords:
   n_word_given_spam = Spam[word].sum() 
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_keywords)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = Ham[word].sum() 
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_keywords)
   parameters_ham[word] = p_word_given_ham


def predict(subject):
    
    subject=subject.replace("\W"," ")
    subject=subject.lower()
    subject=subject.split()

    percentage_spam_in_message=percentage_Spam
    percentage_ham_in_message=percentage_Ham

    for w in subject:
        if w in parameters_spam:
            percentage_spam_in_message*=parameters_spam[w]

        if w in parameters_ham:
            percentage_ham_in_message*=parameters_ham[w]

    print('P(Spam|Message): ', percentage_spam_in_message)
    print('P(Ham|Message): ',percentage_ham_in_message)

    if percentage_ham_in_message>percentage_spam_in_message:
        print("Ham")
    elif percentage_spam_in_message>percentage_ham_in_message:
        print("Spam")
    else:
        print("Equal Chances")

n=input("Enter email subject \n")
predict(n)




