from nltk.classify import NaiveBayesClassifier
from nltk import word_tokenize
import nltk
import pandas as pd

#Test Training data and create the related word list


pos1 = "Boris Johnson: UK will come up with a solution on Huawei"
pos2 = "Business Matters: UK allows Huawei role in 5G networks"
pos3 = "The iPhone at 10: How the smartphone became so smart"
pos4 = "Apple offers free iPhone 4 cases"
pos5 = "Tech Tent: UK gives Huawei the OK"
pos6 = "iPhone 5 launches with larger screen and 4G access"

neg1 = "Why is Huawei still in the UK?"
neg2 = "Best of Today: Huawei risks"
neg3 = "Apple's latest iPhones 'most likely' to get stolen"
neg4 = "Viewpoint: Apple's iPhone launches no longer excite"
neg5 = "Huawei 'failed to improve UK security standards'"
neg6 = "Senator tells MPs Huawei puts US troops at risk"

word_list =list(set(word_tokenize(pos1) + word_tokenize(pos2) + word_tokenize(pos3) + word_tokenize(pos4) + 
	word_tokenize(pos5) + word_tokenize(pos6)+ word_tokenize(neg1)+word_tokenize(neg2)+word_tokenize(neg3)+
	word_tokenize(neg4)+word_tokenize(neg5)+word_tokenize(neg6)))

#pre-processing, recording the shown word as TRUE

def preprocess(sentence):
	return {word: True for word in sentence.lower().split()}

#provide label for traing data

training_data = [[preprocess(pos1),'pos'],
[preprocess(pos2),'pos'],
[preprocess(pos3),'pos'],
[preprocess(pos4), 'pos'],
[preprocess(pos5), 'pos'],
[preprocess(pos6), 'pos'],
[preprocess(neg1), 'neg'],
[preprocess(neg2), 'neg'],
[preprocess(neg3), 'neg'],
[preprocess(neg4), 'neg'],
[preprocess(neg5), 'neg'],
[preprocess(neg6), 'neg']]

#train data with NaiveBayesClassifier
model = NaiveBayesClassifier.train(training_data)

test_pos1 = "iPhone 12 is not great"
test_pos2 = "Huawei"

test_neg1 = "Bent iPhone claims put Apple under pressure to respond"
test_neg2 = "Why Huawei's Google woes worry Africa"

print(test_pos1, '\n', model.classify(preprocess(test_pos1)))
print(test_pos2,'\n' , model.classify(preprocess(test_pos2)))
print(test_neg1,'\n' , model.classify(preprocess(test_neg1)))
print(test_neg2,'\n' , model.classify(preprocess(test_neg2)))

#read news title .csv and use the training model to see the result

data = pd.read_csv('D:/dissertation_program/dissertation/program/result/1/news_title.csv')

sentences = data['titles']

column_number = len(sentences)

sentiment_analysis = [[] for i in range (0, column_number)]

for i in range(0, column_number):
	try:
		sentence = sentences[i]
		sentiment = model.classify(preprocess(sentence))
		sentiment_analysis[i] = sentiment
	except:
		print(i)

Data_result = pd.DataFrame({
	'titles' : sentences,
	'sentiment' :sentiment_analysis
	})

Data_result.to_csv('D:/dissertation_program/dissertation/program/result/1/news_title_result.csv')

