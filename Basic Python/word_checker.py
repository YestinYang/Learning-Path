import urllib

def read_txt(dir):
    quotes = open(dir)
    contents_of_file = quotes.read()
    print (contents_of_file)
    quotes.close()
    return contents_of_file

def checker(text_to_check):
    get_url = urllib.urlopen("http://www.wdylike.appspot.com/?q="+text_to_check)
    output = get_url.read()
    if "true" in output:
        print ("Profanity warning!")
    else:
        print ("No problem!")
    get_url.close()

checker(read_txt(r"D:\movie_quotes.txt"))






