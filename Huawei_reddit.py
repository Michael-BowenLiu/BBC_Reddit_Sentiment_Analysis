from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as WDW
from urllib.parse import quote
from pyquery import PyQuery
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy
import nltk

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
driver = webdriver.Chrome("D:\\Python_try\\chromedriver\\chromedriver.exe", options = chrome_options)

driver.get("https://www.reddit.com/r/Huawei")
time.sleep(3)

js = "return action=document.body.scrollHeight"
height = driver.execute_script(js)

driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
time.sleep(5)

t1 = int(time.time())

status = True

num = 0

while status:
	t2 = int(time.time())
	if t2-t1<800:
		new_height = driver.execute_script(js)
		if new_height>height:
			time.sleep(1)
			driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
			height = new_height
			t1 = int(time.time())
		elif num<5:
			time.sleep(2)
			num = num+1
		else:
			print('SCROLLTOMAX')
			status = False
			driver.execute_script('window.scrollTo(0, 0)')
			break

def url_comments(url):


	item = url
	headers = {'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}


	html = requests.get(item,
	headers=headers)

	soup = BeautifulSoup (html.text, 'lxml')

	comment_1 = [[] for i in soup.select('._2M2wOqmeoPVvcSsJ6Po9-V ._292iotee39Lmt0MkQZ2hPV')]
	datetime_1 = [[] for i in soup.select('._3yx4Dn0W3Yunucf5sVJeFU')]
	comment_pd = pd.DataFrame({
		"comment": comment_1,
		"time": datetime_1
		})


	for i in range(0,1000):
		try:
			text = soup.select('._2M2wOqmeoPVvcSsJ6Po9-V ._292iotee39Lmt0MkQZ2hPV')[i].text
			time = soup.select('._3yx4Dn0W3Yunucf5sVJeFU')[i].text
			comment_1[i] = text
			datetime_1[i] = time
		except:
			break

	comment_pd_1 = pd.DataFrame({
		"comment": comment_1,
		"time": datetime_1
		})

	return comment_pd_1

url = '0'
comment_0 = [[]for i in range(1)]
datetime_0 = [[] for i in range(1)]
comment_data = pd.DataFrame({
	"comment": comment_0,
	"time": datetime_0
	})

for link in driver.find_elements_by_xpath("//a[contains(@href, 'comments')]"):
	new_url = link.get_attribute('href')
	if new_url != url:
		try:
			comment_data_1 = url_comments(new_url)
			comment_data = pd.concat(
				[comment_data, comment_data_1], names = ['comment','time'])
			url = new_url
		except:
			print(new_url)

comment_data.to_csv('Huawei_comments.csv')