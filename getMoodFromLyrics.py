# getMoodFromLyrics.py
# by nino
# ある楽曲の歌詞をスクレイピングしてきて，その曲の印象を予測するプログラム

from bs4 import BeautifulSoup #スクレイピング用のライブラリ
import urllib.request
from urllib.request import Request, urlopen
import sys
import pandas as pd #CSVの扱いが楽になるライブラリ
import time
import re #正規表現ライブラリ
 

#url = 'https://www5.atwiki.jp/hmiku/pages/249.html'

g_maxIndex = 10
g_songDict = {}

class Song:
	# 曲情報を持つクラス
	m_name = ""
	m_url = ""
	m_mood = ""
	m_lyric = ""
	m_bExist = True

	def __init__(self, name, url):
		self.m_name = name # 曲名
		self.m_url = url # 歌詞のURL

	# 歌詞のURLから歌詞を取ってくる
	def setLyric(self):
		req = Request(self.m_url, headers={'User-Agent': 'Mozilla/5.0'})
		response = urlopen(req)
		html = response.read()
		self.soup = BeautifulSoup(html, "lxml")
		orgLyrics = self.soup.find("div", class_="medium")
		# ルビの記述は余計なので除去する
		subLyrics = re.sub('<span class="rt">(.*?)</span>', "", str(orgLyrics))
		subLyrics = re.sub('          ', "",str(subLyrics))
		soup2 = BeautifulSoup(subLyrics, "lxml")
		self.m_lyric = soup2.get_text()
		search_result = re.search(r"調整中です。",self.m_lyric)
		if( search_result != None ): self.m_bExist &= False

	def setMood(self):
		# 最も投票されている印象を取得する
		moods_list = {"yujo","kando","rennai","gennki"}
		for mood in moods_list:
			className = self.soup.find("button", class_="voteBtn mostVoted " + mood + "Btn")
			if( className != None ):
				self.m_mood = mood
		if( self.m_mood == "" ): self.m_bExist &= False


	def showInfo(self):
		print("曲名:" + self.m_name)
		print("URL:" + self.m_url)
		print("印象:" + self.m_mood)
		print("歌詞:" + self.m_lyric)

	def saveCSV(self):
		df = pd.DataFrame(
			[[self.m_name, self.m_url, self.m_mood, self.m_lyric]],
			columns = ['title', 'url', 'mood','lyric']
			)
		df.to_csv('songs/' + self.m_name +'.csv')

	def loadCSV(self, fileName):
		df = pd.read_csv(fileName)
		self.m_name = df.ix[0,"title"]
		self.m_url = df.ix[0,"url"]
		self.m_mood = df.ix[0,"mood"]
		self.m_lyric = df.ix[0,"lyric"]


def getSongFromRanking():

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
				time.sleep(5)
				# print(song_title)
				song_url = "http://utaten.com" + h3.a.get("href")
				song = Song(song_title, song_url)
				song.setLyric()
				song.setMood()
				song.showInfo()
				if( song.m_bExist ):
					song.saveCSV()
					song_dict.update({song_title:song})

			except: 
				pass

	# 辞書型のsong_dictを返す
	return song_dict


def main():
	# 曲をランキングから取得してきてその歌詞を保存する

	#g_songDict = getSongFromRanking()


	#単一の曲を取得するサンプル
	
	song = Song("","")
	song.loadCSV("songs/恋の魔法.csv")
	
	"""
	song = Song("I'll''Bee there","http://utaten.com/lyric/sumika/Summer+Vacation/#sort=popular_sort_asc")
	song.setLyric()
	song.setMood()
	song.showInfo()
	print(song.m_bExist)
	"""
	

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

if __name__ == '__main__':
	main()
	