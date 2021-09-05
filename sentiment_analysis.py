from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import csv

# huawei_data = pd.read_csv('D:/dissertation/program/Huawei_All_800_2.csv')
# iPhone_data = pd.read_csv('D:/dissertation/program/iPhone_All_800.csv')
# xiaomi_data = pd.read_csv('D:/dissertation/program/xiaomi_800_3.csv')
# huawei_sentences = huawei_data['comment']
# iPhone_sentences = iPhone_data['comment']
# xiaomi_sentences = xiaomi_data['comment']

# column_number = huawei_data.shape[0]
# neg = [[] for i in range(0, column_number)]
# neu = [[] for i in range(0, column_number)]
# pos = [[] for i in range(0, column_number)]
# compound = [[] for i in range(0, column_number)]

# print(len(sentences))
# print(column_number)

# for i in range(0, column_number):
# 	sentence = sentences[i]
# 	ss = sia.polarity_scores(sentence)
# 	neg[i] = ss['neg']
# 	neu[i] = ss['neu']
# 	pos[i] = ss['pos']
# 	compound[i] = ss['compound']

# huawei = pd.DataFrame({
# 	'sentence' : sentences,
# 	'neg' : neg,
# 	'neu' : neu,
# 	'pos' : pos,
# 	'compound' : compound
# })

# huawei.to_csv('D:/dissertation/program/Huawei_sentiment.csv', index = False, header = True)
# print(huawei[:10])

def get_sentiment(sentences):
	sia = SentimentIntensityAnalyzer()
	column_number = len(sentences)
	neg = [[] for i in range(0, column_number)]
	neu = [[] for i in range(0, column_number)]
	pos = [[] for i in range(0, column_number)]
	compound = [[] for i in range(0, column_number)]
	for i in range(0, column_number):
		sentence = sentences[i]
		ss = sia.polarity_scores(sentence)
		neg[i] = ss['neg']
		neu[i] = ss['neu']
		pos[i] = ss['pos']
		compound[i] = ss['compound']
	sentiment_result = pd.DataFrame({
		'sentence' : sentences,
		'neg' : neg,
		'neu' : neu,
		'pos' : pos,
		'compound' : compound
	})

	return sentiment_result

for i in ('Huawei','iPhone','xiaomi'):
	data = pd.read_csv('D:/dissertation/program/result/{}_comments.csv'.format(i))
	data_sentences = data['comment']
	data_sentiment = get_sentiment(data_sentences)
	print(i)
	print(data_sentiment[:10])
	data_sentiment.to_csv('D:/dissertation/program/result/{}_reddit_sentiment.csv'.format(i), index = False)

for i in ('Huawei','iPhone','Xiaomi'):
	data = pd.read_csv('D:/dissertation/program/result/{}_news.csv'.format(i))
	data_sentences = data['title']
	data_sentiment = get_sentiment(data_sentences)
	print(i)
	print(data_sentiment[:10])
	data_sentiment.to_csv('D:/dissertation/program/result/{}_news_sentiment.csv'.format(i), index = False)