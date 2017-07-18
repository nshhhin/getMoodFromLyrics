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

class Song:
	# 曲情報を持つクラス

	def __init__(self, name, url, mood, lyric):
		self.m_name = name # 曲名
		self.m_url = url # 歌詞のURL
		self.m_mood = mood # 印象 yujo/kando/rennai/gennki
		self.m_lyric = lyric # 歌詞

	# 歌詞のURLから歌詞を取ってくる
	def getLyric(self):
		# スクレイピングできないページもあるので，FireFoxでアクセスする 
		req = Request(self.m_url, headers={'User-Agent': 'Mozilla/5.0'})
		response = urlopen(req)
		html = response.read()
		soup = BeautifulSoup(html, "lxml")
		orgLyrics = soup.find_all("div", class_="medium")

		# 最も投票されている印象を取得する
		moods_list = {"yujo","kando","rennai","gennki"}
		for i in range(0,len(moods_list)):
			className = soup.find_all("button", class_="voteBtn voted " + moods_list[i])
			if( className != null ):
				mood = mood_list[i]

		# ルビの記述は余計なので除去する
		# <span class="rt">まいばん</span>
		print(orgLyrics)

	def showInfo(self):
		print("曲名:" + self.m_name)
		print("URL:" + self.m_url)
		print("印象:" + self.m_mood)
		print("歌詞:" + self.m_lyric)



def getLyrics():

	for i in range(1, g_maxIndex):

		url = "http://utaten.com/lyricPvRanking/index?page=" + str(i)

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
				
				# print(song_title)
				song_url = "http://utaten.com" + h3.a.get("href")
				song = Song(song_title, song_url, "yujo", "I have a pen")
				song.showInfo()
				song.getLyric()
				song_dict.update({song_title:song})

			except: 
				pass

	# 辞書型のsong_dictを返す
	return song_dict

if __name__ == '__main__':
	# 曲をランキングから取得してきてその歌詞を保存する

	#g_songDict = getLyrics() #曲名と歌詞があるリンク先を取得してくる
	#for _dict in g_songDict:
	#	print( _dict ) #曲名
	#	print( g_songDict[_dict] ) #URL

	# 保存したデータから印象分析をする

	song = Song("JAM LADY","http://utaten.com/lyric/%E9%96%A2%E3%82%B8%E3%83%A3%E3%83%8B%E2%88%9E/JAM+LADY/#sort=popular_sort_asc","gennki","")
	song.getLyric()

	"""
	# wikiから持ってくるバージョン
	# 曲リストを取得してくる
	# 正解印象データはどうするかで詰んだ
	for initial_word in soup.find_all("div", class_="plugin_list_by_tag"):
		for song in initial_word.ul.find_all("li"):
			print(song.a.get_text())
			print(song.a.get("href"))
			song_dict.update({song.a.get_text():song.a.get("href")})

	print(song_dict["結晶"])

	"""