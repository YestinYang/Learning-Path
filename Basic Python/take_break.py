import time
import webbrowser

total_break = 3
current_break = 0

print ("The program starts on" + time.ctime())
while current_break < total_break:
    time.sleep(10)
    webbrowser.open("https://www.youtube.com/watch?v=RP5VqPt_c38&list=LLKwVgBx1nuNXvzoWcUTxmow&t=0s&index=12")
    current_break += 1