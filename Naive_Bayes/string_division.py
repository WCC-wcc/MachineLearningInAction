import re

mySent = 'This book is the best on Python or M.L. I have ever laid eyes upon.'

# result0 = mySent.split()
# print(result0)

# regEx = re.compile('\\W')
# result1 = regEx.split(mySent)
# # print(result1)

# new_result = [tok.lower() for tok in result1 if len(tok) > 0] 


# new_result = []
# for tok in result:
#     if len(tok) > 0:
#         new_result.append( tok.lower() )   #.lower()将字符串全部转换成小写，.upper()将字符串全部转换成大写

# print(new_result)

def textParse(bigString):
	listOfTokens = re.split(r'\W',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 0]

result2 = textParse(mySent)
print(result2)