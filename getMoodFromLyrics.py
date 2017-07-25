#!/usr/bin/env python
# -*- coding: utf-8 -*-
# getMoodFromLyrics.py
# by nino
# ある楽曲の歌詞をスクレイピングしてきて，その曲の印象を予測するプログラム
# 参考：単語ベクトルを作る
# http://yut.hatenablog.com/entry/20120904/1346715336

from bs4 import BeautifulSoup #スクレイピング用のライブラリ
import urllib.request
from urllib.request import Request, urlopen
import sys
import pandas as pd #CSVの扱いが楽になるライブラリ
import time
import re #正規表現ライブラリ
import glob #パス名を見つけたりparseしたりするモジュール
import MeCab
import collections

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm, naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search


g_maxIndex = 10 # 持ってくるページ数
g_maxNumOfSongs = 50 # それぞれの印象に対して持ってくる楽曲数
g_songDict = {}

class Song:
	# 曲情報を持つクラス
	m_name = ""		# 名前
	m_url = ""		# 歌詞が載っているURL
	m_mood = ""		# 印象
	m_lyric = ""	# 歌詞
	m_bExist = True	# 歌詞と印象がしっかりと存在するか

	def __init__(self, name, url):
		self.m_name = name 
		self.m_url = url

	# 歌詞のURLから歌詞を取ってくる
	def setLyric(self,soup):
		"""
		req = Request(self.m_url, headers={'User-Agent': 'Mozilla/5.0'})
		response = urlopen(req)
		html = response.read()
		self.soup = BeautifulSoup(html, "lxml")
		"""

		orgLyrics = soup.find("div", class_="medium")
		# ルビの記述は余計なので除去する
		subLyrics = re.sub('<span class="rt">(.*?)</span>', "", str(orgLyrics))
		subLyrics = re.sub('          ', "",str(subLyrics))
		soup2 = BeautifulSoup(subLyrics, "lxml")
		self.m_lyric = soup2.get_text()

		# 場合によっては歌詞が用意されていない場合もあるので,それをはじく処理
		search_result = re.search(r"調整中です。",self.m_lyric)
		if( search_result != None ): self.m_bExist &= False

	# 印象を取ってくる
	def setMood(self,soup):
		# 最も投票されている印象を取得する
		moods_list = {"yujo","kando","rennai","gennki"}
		for mood in moods_list:
			className = soup.find("button", class_="voteBtn mostVoted " + mood + "Btn")
			if( className != None ):
				self.m_mood = mood
		# 場合によっては印象が投票されていない場合もあるので,それをはじく処理
		if( self.m_mood == "" ): self.m_bExist &= False

	# 歌詞と印象をセットするメソッド
	def setInfo(self):
		req = Request(self.m_url, headers={'User-Agent': 'Mozilla/5.0'})
		response = urlopen(req)
		html = response.read()
		soup = BeautifulSoup(html, "lxml")
		self.setLyric(soup)
		self.setMood(soup)

	# 曲情報を表示するメソッド
	def showInfo(self):
		print("曲名:" + self.m_name)
		print("URL:" + self.m_url)
		print("印象:" + self.m_mood)
		print("歌詞:" + self.m_lyric)

	# 曲情報をCSVとして保存するメソッド
	def saveCSV(self):
		df = pd.DataFrame(
			[[self.m_name, self.m_url, self.m_mood, self.m_lyric]],
			columns = ['title', 'url', 'mood','lyric']
			)
		df.to_csv('songs2/' + self.m_name +'.csv')

	# 曲情報をが書かれたCSVを読み込むメソッド
	def loadCSV(self, fileName):
		df = pd.read_csv(fileName)
		self.m_name = df.ix[0,"title"]
		self.m_url = df.ix[0,"url"]
		self.m_mood = df.ix[0,"mood"]
		self.m_lyric = df.ix[0,"lyric"]

g_bCompleted = False


# 歌詞サイトのランキングから上位の曲の情報をスクレイピングしてくる関数
def getSongFromRanking():
	global g_bCompleted
	i = 1
	gennki_count = 0
	rennai_count = 0
	kando_count = 0
	yujo_count = 0
	while g_bCompleted == False:
		url = "http://utaten.com/lyricPvRanking/index?page=" + str(i)
		i = i + 1
		print(i)
		# FireFoxでアクセスする 
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
				song_url = "http://utaten.com" + h3.a.get("href")
				song = Song(song_title, song_url)
				song.setInfo()
				print(song.m_name)

				if( song.m_bExist ):
					if( song.m_mood == "gennki" and gennki_count < 50 ):
						gennki_count = gennki_count + 1
						song.saveCSV()
						song_dict.update({song_title:song})
					elif( song.m_mood == "rennai" and rennai_count < 50 ):
						rennai_count = rennai_count + 1
						song.saveCSV()
						song_dict.update({song_title:song})
					elif( song.m_mood == "kando" and kando_count < 50 ):
						kando_count = kando_count + 1
						song.saveCSV()
						song_dict.update({song_title:song})
					elif( song.m_mood == "yujo" and yujo_count < 50 ):
						yujo_count = yujo_count + 1
						song.saveCSV()
						song_dict.update({song_title:song})
					
					if(  gennki_count == 50 and rennai_count == 50 and kando_count == 50 and yujo_count == 50):
						# 全部カウントされなくなったら
						g_bCompleted = True

					print("元気" + str(gennki_count))
					print("恋愛" + str(rennai_count))
					print("感動" + str(kando_count))
					print("友情" + str(yujo_count))

			except: 
				pass

	# 辞書型のsong_dictを返す
	return song_dict


def word_counter(words):
    count_dict = collections.Counter(words) #出現頻度を数える
    return dict(count_dict)

def get_unigram(file_path):
    result = []
    python_version = sys.version_info.major
    
    if python_version >= 3:
        for file in file_path:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    count_dict = word_counter(line)
                    result.append(count_dict)
    else:
        for file in file_path:
            with open(file, 'r') as f:
                for line in f:
                    count_dict = word_counter(line)
                    result.append(count_dict)
    
    return result


def train_model(features, labels, method='SVM', parameters=None):
    ### set the model
    if method == 'SVM':
        model = svm.SVC()
    elif method == 'NB':
        model = naive_bayes.GaussianNB()
    elif method == 'RF':
        model = RandomForestClassifier()
    else:
        print("Set method as SVM (for Support vector machine), NB (for Naive Bayes) or RF (Random Forest)")
    ### set parameters if exists
    if parameters:
        model.set_params(**parameters)
    ### train the model
    model.fit( features, labels )
    ### return the trained model
    return model

def N_splitter(seq, N):
    avg = len(seq) / float(N)
    out = []
    last = 0.0
    
    while last < len(seq):
        out.append( seq[int(last):int(last + avg)] )
        last += avg
        
    return np.array(out)



def predict(model, features):
    predictions = model.predict( features )
    return predictions

def evaluate_model(predictions, labels):
    data_num = len(labels)
    correct_num = np.sum( predictions == labels )
    return data_num, correct_num

def cross_validate(n_folds, feature_vectors, labels, shuffle_order, method='SVM', parameters=None):
    result_test_num = []
    result_correct_num = []
    
    print( "おおおおおおお" )
    print( len(labels) )
    n_splits = N_splitter( range(len(labels)), n_folds )


    for i in range(n_folds):
        print( "Executing {0}th set...".format(i+1) )
        print(n_splits[i])
        
        test_elems = shuffle_order[ n_splits[i] ]
        train_elems = np.array([])
        train_set = n_splits[ np.arange(n_folds) !=i ]
        for j in train_set:
            train_elems = np.r_[ train_elems, shuffle_order[j] ]
        train_elems = train_elems.astype(np.integer)
        print( "=====" )
        print( train_elems )

        # train
        model = train_model( feature_vectors[train_elems], labels[train_elems], method, parameters )
        # predict
        predictions = predict( model, feature_vectors[test_elems] )
        # evaluate
        test_num, correct_num = evaluate_model( predictions, labels[test_elems] )
        result_test_num.append( test_num )
        result_correct_num.append( correct_num )
    
    return result_test_num, result_correct_num

def main():

	# 曲をランキングから取得してきてその歌詞を保存する
	# (何度もやってると怒られそうなので、必要な時しかしないでください)
	g_songDict = getSongFromRanking()

	#単一の曲を取得するサンプル
	# song = Song("","")
	# song.loadCSV("songs/恋の魔法.csv")
	
	mecab = MeCab.Tagger("-Ochasen") # MeCabオブジェクトを生成

	"""
		parse = mecab.parseToNode( song.m_lyric )
		
		words = []
		while parse:
			print( parse.surface )
			words.append( parse.surface )
			parse = parse.next
		
		wordsCount_dict = word_counter( words )

		unigram = []

		print(wordsCount_dict)
	"""

	#songsディレクトリに保存したファイル全てを読み込む
	songsPath = glob.glob("songs2/*")

	gennki_count = 0
	rennai_count = 0
	kando_count = 0
	yujo_count = 0

	result = []
	labels = []

	for songPath in songsPath:
		song = Song("","")
		song.loadCSV(songPath)
		song.showInfo()
		
		if song.m_mood == "gennki":
			gennki_count = gennki_count + 1
			labels.append(0)
		elif song.m_mood == "rennai":
			rennai_count = rennai_count + 1
			labels.append(1)
		elif song.m_mood == "kando":
			kando_count = kando_count + 1
			labels.append(2)
		elif song.m_mood == "yujo":
			yujo_count = yujo_count + 1
			labels.append(3)


		parse = mecab.parseToNode( song.m_lyric )
		words = []
		while parse:
			print( parse.surface )
			words.append( parse.surface )
			parse = parse.next

		wordsCount_dict = word_counter( words )

		result.append( wordsCount_dict )

	"""
	vec = DictVectorizer()
	feature_vectors_csr = vec.fit_transform( result )
	print( feature_vectors_csr )

	one_third_size = int(len(songsPath)/3)

	np.random.seed(7789)
	shuffle_order = np.random.choice( len(songsPath), len(songsPath), replace=False )

	search_parameters = [
    	{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000]},
    	{'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}
	]

	model = svm.SVC()
	clf = grid_search.GridSearchCV(model, search_parameters)
	clf.fit( feature_vectors_csr, labels ) 

	N_FOLDS = 3
	ans, corr = cross_validate(N_FOLDS, feature_vectors_csr, labels, shuffle_order, method='SVM', parameters=clf.best_params_)

	print( "average precision : ", np.around( 100.*sum(corr)/sum(ans), decimals=1 ), "%" )
	"""

	print("元気" + str(gennki_count))
	print("恋愛" + str(rennai_count))
	print("感動" + str(kando_count))
	print("友情" + str(yujo_count))

	"""
	song = Song("I'll''Bee there","http://utaten.com/lyric/sumika/Summer+Vacation/#sort=popular_sort_asc")
	song.setLyric()
	song.setMood()
	song.showInfo()
	print(song.m_bExist)

	# wikiから持ってくるバージョン
	# 曲リストを取得してくる
	# 正解印象データはどうするかで詰んだ
	for initial_word in soup.find_all("div", class_="plugin_list_by_tag"):
		for song in initial_word.ul.find_all("li"):
			print(song.a.get_text())
			print(song.a.get("href"))
			song_dict.update({song.a.get_text():song.a.get("href")})

	print(song_dict["結晶"]
	"""



if __name__ == '__main__':
	main()
	