import json, os, sys, pdb
import pprint
import PIL, requests
from StringIO import StringIO
from PIL import Image
dir_path = '/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data'
datasets = ['BuzzFeed', 'PolitiFact']
news = ['FakeNewsContent', 'RealNewsContent']
top_img_dir =  './top_imgs'
https_proxy = {"http":"http://10.237.170.39:8080", "https":"https://10.237.170.39:8080"}

i = 0
n1, n2, N1, N2 = 0, 0, 0, 0
for dataset in datasets:
	dir = os.path.join(dir_path, dataset)
	for news_type in news:
		dir1 = os.path.join(dir, news_type)
		print dir1
		for File in os.listdir(dir1):
			file_path = os.path.join(dir1, File)
			f = open(file_path)
			try:
				data = json.load(f)
			except Exception as e:
				continue
			img_name = data['top_img']
			#images = data['images']
			#N = len(images) + 1
			try:
				response = requests.get(img_name, proxies=https_proxy)
				#print response
			except Exception as e:
				#print e
				pass
				#continue
			try:
				img = Image.open(StringIO(response.content))
				print(img)
				if i == 0:
					n1 += 1
				else:
					n2 += 1
			except Exception as e:
				#print e
				pass
			if i == 0:
				N1 += 1
			else:
				N2 += 1
			#for Img in images:
			#	try:
			#		response = requests.get(Img, proxies=https_proxy)
			#		img = Image.open(StringIO(response.content))
			#		print(img)
			#		n += 1
			#	except Exception as e:
			#		print e
	i += 1


print "For 1st dataset, stats are:"
print "Total %d, missed %d"%(n1, N1)
print "For 2nd dataset, stats are:"
print "Total %d, missed %d"%(n2, N2)
