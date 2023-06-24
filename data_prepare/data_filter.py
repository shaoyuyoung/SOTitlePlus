"""
Used to clean data from dumped logs
"""
from lxml import etree
import pandas as pd
from bs4 import BeautifulSoup
from itertools import islice
import numpy as np
from tqdm import *
import re

lans = ['python', 'java', 'javascript', 'c#', 'html', 'php']
for lan in lans:
    to = pd.DataFrame(columns=['Id', 'Body', 'Code', 'Title', 'Score', 'Tags',
                               'accepted'])
    print("start::::", lan)
    context = etree.iterparse('Posts.xml', events=('end',), tag='row')
    lanTag = "<" + lan + ">"
    bar = context
    CodeList = []
    BodyList = []
    TitleList = []
    ScoreList = []
    IdList = []
    Tags = []
    Body, Title, to = None, None, None
    for event, element in tqdm(bar):
        temp = element.attrib
        body, score, title, tag, accepted, postTypeId, Id, discription, code = '', 0, '', '', 0, 0, 0, '', ''

        for key in temp:
            if key == "PostTypeId":
                postTypeId = temp[key]
                if postTypeId != '1':
                    break
            elif key == "Id":
                Id = temp[key]
            elif key == "Body":
                body = temp[key]
            elif key == "Score":
                score = temp[key]
                score = int(score)
            elif key == "Title":
                title = temp[key]
            elif key == "Tags":
                tag = temp[key]
            elif key == "AcceptedAnswerId":
                accepted = temp[key]
                accepted = int(accepted)
        if (score >= 10) and postTypeId == '1' and (lanTag in tag) and accepted != 0 and (
                "</code></pre>" in body):
            body = re.sub('[^a-zA-Z0-9~!@#$%^&*()_+`\-={}\[\]|\\\:";\'<>?,./ ]', "", body)
            title = re.sub('[^a-zA-Z0-9~!@#$%^&*()_+`\-={}\[\]|\\\:";\'<>?,./ ]', "", title)
            pattern = re.compile(r'(.*?)<pre(.*?)</pre>', re.S)
            discription = ''
            code = ''
            while (len(body)):
                match = re.match(pattern, body)
                if match:
                    discription = discription + ' ' + match.group(1)
                    code = code + ' ' + '<pre' + match.group(2) + '</pre>'
                    body = body[match.end():]
                else:
                    discription = discription + ' ' + body
                    body = ""

            discription = BeautifulSoup(discription, 'html.parser').get_text()
            code = BeautifulSoup(code, 'html.parser').get_text()


            discription = discription.replace('\n', " ")
            discription = re.sub(" +", " ", discription)[1:]
            code = code.replace('\n', ' ')
            code = re.sub(" +", " ", code)[1:]

            discription = discription.encode('utf-8', 'ignore').decode()
            code = code.encode('utf-8', 'ignore').decode()
            title = title.encode('utf-8', 'ignore').decode()
            tag = tag.encode('utf-8', 'ignore').decode()

            to = pd.DataFrame([[Id, discription, code, title, score, tag, accepted]],
                              columns=['Id', 'Body', 'Code', 'Title', 'Score', 'Tags', 'accepted'])
        element.clear()
