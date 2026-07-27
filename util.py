

def writeToReport(path,content):
    f= open(path,"a")
    f.write(content+'\n')
    f.close()

def list_to_str(list):
    str_list = ''
    for l in list:
        #print(l)
        str_list += str(l) + ','
    return str_list