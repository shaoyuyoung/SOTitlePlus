from lxml import etree
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import *
import re

LANGUAGES = ['python', 'java', 'c#', 'html', 'go', 'ruby', 'javascript', 'php']
XML_FILE_PATH = 'Posts.xml'

if __name__ == "__main__":
    for lan in LANGUAGES:
        save_file = f"{lan}.csv"
        to = pd.DataFrame(columns=['Id', 'Body', 'Code', 'Title', 'Score', 'Tags',
                                   'accepted'])
        to.to_csv(save_file, header=True, index=False)
        print(f"{lan} starts!")
        context = etree.iterparse(XML_FILE_PATH, events=('end',), tag='row')
        lanTag = "<" + lan + ">"
        bar = tqdm(context)
        CodeList = []
        BodyList = []
        TitleList = []
        ScoreList = []
        IdList = []
        Tags = []
        Body, Title, to = None, None, None
        for event, element in bar:
            temp = element.attrib
            body, score, title, tag, accepted, postTypeId, Id, description, code = '', 0, '', '', 0, 0, 0, '', ''

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
            if score >= 10 and postTypeId == '1' and lanTag in tag and accepted != 0 and "</code></pre>" in body:
                # remove Mojibake
                body = re.sub('[^a-zA-Z0-9~!@#$%^&*()_+`\-={}\[\]|\\\:";\'<>?,./ ]', "", body)
                title = re.sub('[^a-zA-Z0-9~!@#$%^&*()_+`\-={}\[\]|\\\:";\'<>?,./ ]', "", title)
                # extract description and code
                pattern = re.compile(r'(.*?)<pre(.*?)</pre>', re.S)
                description = ''
                code = ''
                while len(body):
                    match = re.match(pattern, body)
                    if match:
                        description = description + ' ' + match.group(1)
                        code = code + ' ' + '<pre' + match.group(2) + '</pre>'
                        body = body[match.end():]
                    else:
                        description = description + ' ' + body
                        body = ""
                # remove markets
                description = BeautifulSoup(description, 'html.parser').get_text()
                code = BeautifulSoup(code, 'html.parser').get_text()

                # clap
                description = description.replace('\n', " ")
                description = re.sub(" +", " ", description)  # merge space
                code = code.replace('\n', ' ')
                code = re.sub(" +", " ", code)  # merge space

                # Encode and decode, convert to utf-8
                description = description.encode('utf-8', 'ignore').decode().strip()
                code = code.encode('utf-8', 'ignore').decode().strip()
                title = title.encode('utf-8', 'ignore').decode()
                tag = tag.encode('utf-8', 'ignore').decode()

                to = pd.DataFrame([[Id, description, code, title, score, tag]],
                                  columns=['Id', 'Desc', 'Code', 'Title', 'Score', 'Tags'])
                to.to_csv(save_file, header=False, index=False, mode='a+')
            element.clear()
