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

for i in range (1,30):
	try:
		item = 'https://www.bbc.co.uk/search?q=huawei&page={}'.format(i)
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

title_pd = title_pd[title_pd['title'].str.contains("Huawei")]

print(title_pd)

title_pd.to_csv('D:/dissertation/program/result/huawei_news.csv', index = False)

# for i in range (0,2):
# 	try:
# 		item = 'https://www.google.com/search?q=iPhone&tbm=nws&start={}'.format(i*10)
# 		html = requests.get(item,
# 			headers=headers)
# 		soup = BeautifulSoup (html.text, 'lxml')
# 		for j in range (0,10):
# 			title = soup.select('.JheGif .nDgy9d')[j].text
# 			datetime = soup.select('.wxp1Sb')[j].text
# 			# title_1[j] = title
# 			# datetime_1[j] = datetime
# 			print(title)
# 			print(datetime)
# 	except:
#  		break

# comment_1 = [[] for i in soup.select('._2M2wOqmeoPVvcSsJ6Po9-V ._292iotee39Lmt0MkQZ2hPV')]
# datetime_1 = [[] for i in soup.select('._3yx4Dn0W3Yunucf5sVJeFU')]
# comment_pd = pd.DataFrame({
# 	"comment": comment_1,
# 	"time": datetime_1
# 	})


# for i in range(0,25):
# 	try:
# 		text = soup.select('._2M2wOqmeoPVvcSsJ6Po9-V ._292iotee39Lmt0MkQZ2hPV')[i].text
# 		time = soup.select('._3yx4Dn0W3Yunucf5sVJeFU')[i].text
# 		comment_1[i] = text
# 		datetime_1[i] = time
# 		# comment_pd_1 = pd.DataFrame({
# 		# 	"comment" : text,
# 		# 	"time" : time
# 		# 	})
# 		# comment_pd = pd.concat(
# 		# 	[comment_pd, comment_pd_1], names = ['comment','time'])
# 		print(text)
# 		print(time)
# 	except:
# 		break

# comment_pd_1 = pd.DataFrame({
# 	"comment": comment_1,
# 	"time": datetime_1
# 	})

# print(comment_pd_1)
# 	# text = soup.select('.content .md')[2].text

# 	# print(text)


