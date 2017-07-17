# getLyrics2.py
# implemented by nino
# 歌詞をまとめているwikiにアクセスし,スクレイピングする

from bs4 import BeautifulSoup
import urllib.request
from urllib.request import Request, urlopen

import re #正規表現ライブラリ
 

#url = 'https://www5.atwiki.jp/hmiku/pages/249.html'



for()
url = "http://utaten.com/lyricPvRanking/index?page=4"

# スクレイピングできないページもあるので，FireFoxでアクセスする 

req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urlopen(req)
html = response.read()
soup = BeautifulSoup(html, "lxml")

# {曲名:歌詞が載っているURL}の辞書song_dictを作る
song_dict =  {}

for h3 in soup.find_all("h3"):
	#print(h3.a.get_text())
	text = h3.a.get_text()
	split_text = text.split("        ")
	print(split_text[1])


"""
# 曲リストを取得してくる
for initial_word in soup.find_all("div", class_="plugin_list_by_tag"):
	for song in initial_word.ul.find_all("li"):
		print(song.a.get_text())
		print(song.a.get("href"))
		song_dict.update({song.a.get_text():song.a.get("href")})

print(song_dict["結晶"])

"""
'''
<div class="plugin_list_by_tag">
<ul>
~~~~ ←ここをもってくる
</ul>
</div>
'''