import codecs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def h2c():
    filec=codecs.open("/home/ubuntu/zzz/ICD-MSMN-master/change_data/combined_dataset",'r','utf-8')
    combined_code = {}
    line=filec.readline()
    while line:
        # print(line)
        if "start" in line[0:6]:
            hamd_id = line.split("!")[-1].replace('\r\n','')
            line_next = filec.readline()
            if "code" in line_next:
                code = line_next.split(':')[-1].replace('\r\n','').split(' ')
                del code[0]
                del code[-1]
            combined_code[hamd_id] = code
        line = filec.readline()
    return combined_code

def c2wiki():
    code2wiki={}
    file1=codecs.open("/home/ubuntu/zzz/ICD-MSMN-master/change_data/wikipedia_knowledge",'r','utf-8')
    line=file1.readline()
    wiki_note = ''
    logo_code = []
    while line:
        if 'XXXdiseaseXXX ' in line:
            wiki_note = ''
            logo_code = []
            logo = line.split(' ')
            for i in logo:
                if i[0:1] == 'd':
                    logo_code.append(i.replace('\n',''))
        elif line != 'XXXendXXX':
            wiki_note = wiki_note + line.replace('\n',' ')
            # wiki_note = wiki_note + [' ']
            line_next_wiki = file1.readline()
        if 'XXXendXXX' in line:
            # print(line)
            for i in logo_code:
                code2wiki[i] = wiki_note
        line=file1.readline()
    return code2wiki
# print(len(code2wiki))