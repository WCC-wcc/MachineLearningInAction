from tkinter import *
from numpy import*
root = Tk()
myLabel  = Label(root,text='Hello World')
myLabel.grid()

chkBtnVar = IntVar() 																	#IntVar()为复选框是否被选中的标识
chkBtn = Checkbutton(root,text = 'Model Tree',variable = chkBtnVar)	
chkBtn.grid(row = 3,column = 0,columnspan = 2)	
print(IntVar())
root.mainloop()
print(arange(0,10,1))