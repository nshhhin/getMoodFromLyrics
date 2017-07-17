# getLyrics2.py
# implemented by nino
# 歌詞をまとめているwikiにアクセスし,スクレイピングする

from bs4 import BeautifulSoup
import urllib.request
from urllib.request import Request, urlopen
import sys

import re #正規表現ライブラリ
 

#url = 'https://www5.atwiki.jp/hmiku/pages/249.html'

g_maxIndex = 2
g_songDict = {}

def getLyrics():

	for i in range(1,g_maxIndex):

		url = "http://utaten.com/lyricPvRanking/index?page="+str(i)

		# スクレイピングできないページもあるので，FireFoxでアクセスする 

		req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
		response = urlopen(req)
		html = response.read()
		soup = BeautifulSoup(html, "lxml")

		# {曲名:歌詞が載っているURL}の辞書song_dictを作る
		song_dict =  {}

		for h3 in soup.find_all("h3"):
			#print(h3.a.get_text())
			try:
				text = h3.a.get_text()
				# 余分な空白を取り除く
				split_text = text.split("        ")
				song_title = split_text[1]
				song_title = song_title[0:len(song_title)-6]
				print(song_title)
				song_url = "http://utaten.com" + h3.a.get("href")
				song_dict.update({song_title:song_url})
			except: 
				pass
				"      "

	# 辞書型のsong_dictを返す
	return song_dict

g_songDict = getLyrics()
print(g_songDict["青春のすべて"])
#g_songDict.has_key("青春のすべて")

"""
# wikiから持ってくるバージョン
# 曲リストを取得してくる
for initial_word in soup.find_all("div", class_="plugin_list_by_tag"):
	for song in initial_word.ul.find_all("li"):
		print(song.a.get_text())
		print(song.a.get("href"))
		song_dict.update({song.a.get_text():song.a.get("href")})

print(song_dict["結晶"])

"""