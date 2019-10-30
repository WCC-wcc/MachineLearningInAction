import urllib
import requests
import json
def geoGrab(stAddress,city):
	apiStem = 'http://where.yahooapis.com/geocode'
	params = {}
	params['flags'] = 'J'
	params['appid'] = 'ppp68N8t'
	params['location'] = '%s %s' % (stAddress,city)
	url_params = urllib.parse.urlencode(params)
	yahooApi = apiStem + url_params
	print(yahooApi)
	c = urllib.request.urlopen(yahooApi)
	return json.loads(c.read())

print(geoGrab('8666a SW Canyon Road','Beaverton, OR'))


from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    #对文件中的每个样本调用geoGrab()获取json数据,解析后写入源文件
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()