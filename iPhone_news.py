from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy
import nltk



headers = {'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'}

title = []
title_pd = pd.DataFrame({
	'title' : title
	})

for i in range (1,24):
	try:
		item = 'https://www.bbc.co.uk/search?q=iPhone+smartphone&page={}'.format(i)
		html = requests.get(item,
		headers=headers)
		soup = BeautifulSoup (html.text, 'lxml')
		
		title_1 = [[] for i in range(0,10)]

		for j in range (0,10):
			title = soup.select('.e1f5wbog4')[j].text
			title_1[j] = title

		title_pd_1 = pd.DataFrame({
			'title': title_1
			})
		title_pd = pd.concat(
				[title_pd, title_pd_1], names = ['title'])
	except:
		break
#将含有iPhone的表格挑出

title_pd = title_pd[title_pd['title'].str.contains("iPhone")]

print(title_pd)

title_pd.to_csv('D:/dissertation/program/result/iPhone_news.csv', index = False)
