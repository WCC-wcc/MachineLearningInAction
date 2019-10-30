# RSS（Really Simple Syndication，简易信息聚合）。它是一种消息来源格式规范，用以聚合经常发布更新数据的网站，例如博客文章、新闻、音频或视频的网摘。
# RSS文件包含全文或是节录的文字，再加上发布者所订阅之网摘数据和授权的元数据。把新闻标题、摘要（Feed）、内容按照用户的要求，“送”到用户的桌面就是RSS的目的。

import feedparser
rss_docNY = feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')	#纽约	feedparser.parse返回的是个字典。
rss_docSB = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss') 		#旧金山 	feedparser.parse返回的是个字典。
print(len(rss_docNY['entries']))														#rss_doc['entries']是所有帖子，它是个List，
print(len(rss_docSB['entries']))

for entry in rss_docNY['entries']:														#里面每一条entry是一条帖子。每个entry又是个字典，														
	print (entry['summary_detail']['value'])											#entry['summary_detail']是帖子详情， 它也是个字典。																																					
																							
for entry in rss_docSB['entries']:														#里面每一条entry是一条帖子。每个entry又是个字典，	
	print (entry['summary_detail']['value'])											#entry['summary_detail']是帖子详情， 它也是个字典。