import nltk
import os
import difflib, flask
from flask import Flask, render_template, send_from_directory, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager
plt.figure(figsize=(10, 5)) 
matplotlib.pyplot.rc('font', family='SimHei')

import re, difflib
import editdistance


# In[8]:


import reportlab
import os
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image
from reportlab.lib.styles import getSampleStyleSheet

home = os.path.expanduser("~")

pdfmetrics.registerFont(TTFont('MSJH', './fonts/MSJH.TTC'))
pdfmetrics.registerFont(TTFont('MSJHBD', './fonts/MSJHBD.TTC'))
registerFontFamily("MSJH", normal="MSJH", bold="MSJHBD")
FONT_NAME = "MSJH"

class MyCSS:
    h1 = ParagraphStyle(name="h1", fontName=FONT_NAME, fontSize=21, leading=28, alignment=1, spaceAfter=16)
    h3 = ParagraphStyle(name="h3", fontName=FONT_NAME, fontSize=14, leading=21, spaceBefore=16)
    h5 = ParagraphStyle(name="h5", fontName=FONT_NAME, fontSize=12, leading=21, alignment=1, spaceBefore=16)
    p = ParagraphStyle(name="BodyText", fontName=FONT_NAME, fontSize=12, leading=18, spaceBefore=8, firstLineIndent=24)
    r = ParagraphStyle(name="BodyText", fontName=FONT_NAME, fontSize=12, leading=18, spaceBefore=8, firstLineIndent=24, textColor=colors.red)

class PiiPdf:
    @classmethod
    def doH1(cls, text: str):
        return Paragraph(text, MyCSS.h1)

    @classmethod
    def doH3(cls, text: str):
        return Paragraph(text, MyCSS.h3)

    @classmethod
    def doH5(cls, text: str):
        return Paragraph(text, MyCSS.h5)

    @classmethod
    def doP(cls, text: str):
        return Paragraph(text, MyCSS.p)

    @classmethod
    def doLine(cls):
        drawing = Drawing(500, 220)
        line = LinePlot()
        line.x = 50
        line.y = 50
        line.height = 125
        line.width = 300
        line.lines[0].strokeColor = colors.blue
        line.lines[1].strokeColor = colors.red
        line.lines[2].strokeColor = colors.green
        line.data = [((0, 50), (100, 100), (200, 200), (250, 210), (300, 300), (400, 800))]

        drawing.add(line)
        return drawing

    @classmethod
    def doChart(cls, data):
        drawing = Drawing(width=500, height=200)
        pie = Pie()
        pie.x = 150
        pie.y = 65
        pie.sideLabels = False
        pie.labels = [letter for letter in "abcdefg"]
        pie.data = data  # list(range(15, 105, 15))
        pie.slices.strokeWidth = 0.5

        drawing.add(pie)
        return drawing

from reportlab.platypus import TableStyle
table_style=TableStyle([
         ('FONT', (0, 0), (-1, -1), 'MSJH', 10),
         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
         ('GRID', (0,0), (-1,-1), 0.5, colors.black),
         ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
         ('BOX', (0,0), (-1,-1), 0.25, colors.black),
         ('BACKGROUND',(0,0),(-1,-1),colors.white)])
table_style_red = TableStyle([
         ('FONT', (0, 0), (-1, -1), 'MSJH', 10),
         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
         ('GRID', (0,0), (-1,-1), 0.5, colors.black),
         ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
         ('BOX', (0,0), (-1,-1), 0.25, colors.black),
         ('BACKGROUND',(0,0),(-1,-1),colors.lavender)])


# In[9]:


import jieba
import jieba.analyse
jieba.set_dictionary("./dict.txt")
# 載入自定義詞庫：jieba.load_userdict(file_path)
# 加入字詞：jieba.add_word(word, freq=None, tag=None)
# 刪除字詞：jieba.del_word(word)

with open('./stopwords_zhTW.txt', encoding="utf-8") as f:
    stopword_list = [s for s in f.read().split('\n')]


# ###**預處理**
# 1. 移除內文停用詞，Ex: ! “ # $ % &等各種符號。
# 
# 2. 使用結巴套件斷詞 
# 
# 3. 移除空白段落

# In[10]:


def load_file():
    file1 = ''
    with open('./uploads/TEXT1.txt','r', encoding = 'utf-8') as load:
        file1 = load.readlines()
        load.close()
    file2 = ''
    with open('./uploads/TEXT2.txt', 'r', encoding = 'utf-8') as mem:
        file2 = mem.readlines()
    mem.close()
    return file1, file2


# In[11]:


def pre_file1(file1):
    a_part = ''
    for i in range(len(file1)):
        if file1[i] != '\n':
            word_list = jieba.cut(file1[i], cut_all=False)
            for word in word_list:
                if word not in stopword_list:
                    a_part += word + " "
    return a_part


# ###**關鍵字**：TF-IDF，找出的關鍵詞會依照詞頻權重排列

# **全文詞向量**
# 
# 1. 迅速理解文本主題 
# 
# 2. 網路爬蟲關鍵字檢索

# In[12]:


# 全文詞向量：文章主題＆網路爬蟲關鍵字(10)
def all_part(a_part):
    all_part = jieba.analyse.extract_tags(a_part, topK=10, withWeight=False, allowPOS=()) #topK為返回幾個TF / IDF權重最大的關鍵詞，默認值為20
    print(all_part)
    #for x, w in jieba.analyse.extract_tags(a_part, topK=10, withWeight=True):
        #print('%s %s' % (x, w))
    t1, t2, t3 = all_part[0], all_part[1], all_part[2]
    return all_part


# In[13]:


# 詞彙分佈圖
import matplotlib.pyplot as plt
def img(all_part):
    fig = plt.figure(figsize=(10, 5)) 
    raw = open("./TEXT1.txt").read() 
    text1 = nltk.text.Text(jieba.cut(raw))
    graph1 = nltk.text.Text(text1)
    graph1.dispersion_plot(all_part)
    fig.savefig('./output_files/plot.png')


# In[14]:


# 詞彙多樣性：全文
def lexical_diversity(text):
    return len(set(text)) / len(text)


# **段落詞向量**
# 1. 文本對齊的前置作業。
# 
# 2. 計算文章每個段落裡，前三個關鍵字，作為該段落比對的代表，段落之間關鍵字相同時，則可進行段落間的比對。

# In[15]:


# 段落詞向量：文本對齊的關鍵字(3)
def part_(file):
    part = []
    for i in range(len(file)):
        s = file[i]
        epart = jieba.analyse.extract_tags(s, topK=3, withWeight=False, allowPOS=()) #topK為返回幾個TF / IDF權重最大的關鍵詞，默認值為20
        part.append(epart)
        print(epart)
        #for x, w in jieba.analyse.extract_tags(s, topK=3, withWeight=True):
            #print('%s %s' % (x, w))
        print()
    return part


# **句子的交互比對－相似度分析**
# 
# 1. Diff: Difflib文本相似度比對套件的函式
# 
# 2. LCS: 最長公共子串列
# 
# 3. Edit distance: 編輯距離

# In[16]:


def lcs(s1, s2, m, n):
    num = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(1,m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                num[i][j] = num[i-1][j-1]+1
            else:
                num[i][j] = max(num[i-1][j], num[i][j-1])
    return num[-1][-1]


# In[17]:


import re, difflib
import editdistance
def analyze(partA, partB, story, file1, file2, p):
    for ai in range(len(partA)):
        for aj in partA[ai]:
            for bi in range(len(partB)):
                for bj in partB[bi]:
                    if aj == bj:
                        print("【關鍵字】：", aj)
                        sk = "【關鍵字】：" + aj
                        story.append(p.doH5(sk))
                        keyword1 = [['文本1 (段落)', '文本2 (段落)']]
                        keyword2 = [['相似度', 'LCS', '編輯距離']]
                        aList = re.split('，|。', str(file1[ai]))
                        bList = re.split('，|。', str(file2[bi]))
                        for s1 in aList:
                            if aj in s1:
                                for s2 in bList:
                                    if aj in s2:
                                        if len(s1) > 20:
                                            s1 = s1[0:20] + "..."
                                        if len(s2) > 20:
                                            s2 = s2[0:20] + "..."
                                        if str(ai+1) in s1:
                                            s1 = s1
                                        else:
                                            s1 = s1 + "   (" + str(ai+1) + ")"
                                        if str(bi+1) in s2:
                                            s2 = s2
                                        else:
                                            s2 = s2 + "   (" + str(bi+1) + ")"
                                        list1 = []
                                        list2 = []
                                        list1.append(s1)
                                        list1.append(s2)
                                        print("%1s %-50s %1s %-50s" % ("A", s1, "B", s2))  # 相似段落
                                        diff_cal = difflib.SequenceMatcher(None, s1, s2).ratio() #diff_cal
                                        diff_cal = round(diff_cal,2)
                                        list2.append(diff_cal)
                                        print("Diff: ", diff_cal)
                                        lcs_cal = lcs(s1, s2, len(s1), len(s2)) #lcs_cal
                                        list2.append(lcs_cal)
                                        print("LCS: ", lcs_cal)
                                        editDistance = editdistance.eval(s1, s2)
                                        list2.append(editDistance)
                                        print("Edit distance: ", editDistance) #editdistance
                                        print()
                                        if diff_cal >= 0.5:
                                            keyword1.append(list1)
                                            content = Table(keyword1, colWidths=240, style=table_style_red)
                                            story.append(content)
                                            
                                            keyword2.append(list2)
                                            number = Table(keyword2, colWidths=160, style=table_style_red)
                                            story.append(number)
                                        else:
                                            keyword1.append(list1)
                                            content = Table(keyword1, colWidths=240, style=table_style)
                                            story.append(content)
                                            
                                            keyword2.append(list2)
                                            number = Table(keyword2, colWidths=160, style=table_style)
                                            story.append(number)
                                        keyword1 = []
                                        keyword2 = []
    report_add = "/content/drive/My Drive/NProject/Hello_RE.pdf"
    doc = SimpleDocTemplate(report_add)
    doc.build(story)
    return report_add


# In[18]:


UPLOAD_FOLDER = 'C:/Users/yulin/anaconda3/GProject_/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")
@app.route('/Background.png')
def background(): 
    return send_from_directory(os.path.join(app.root_path, '/static'), 'Background.png')
@app.route('/favicon.svg')
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, '/static'), 'favicon.svg')


# In[19]:


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
            
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


# In[20]:


@app.route('/file-compared', methods=['POST', 'GET'])
def compared_file():
    p = PiiPdf()
    story = []
    story.append(p.doH1("<b>比對報告</b>"))
    f1,f2 = load_file()
    story.append(p.doH3("<b>全文關鍵字</b>"))
    a_part = pre_file1(f1)
    b_part = pre_file1(f2)
    allPart = all_part(a_part)
    photo = Image('./output_files/plot.png', width=376, height=273)
    story.append(p.doH3("<b>詞彙分布圖</b>"))
    story.append(photo)
    img(allPart)
    story.append(p.doH3("<b>詞彙多樣性</b>"))
    score1 = lexical_diversity(a_part)
    score2 = lexical_diversity(b_part)
    score = [['文本1', '文本2']]
    score_ = [score1, score1]
    score.append(score_)
    score_F = Table(score, colWidths=220, style=table_style)
    story.append(score_F)
    story.append(p.doH3("<b>段落關鍵字</b>"))
    partA = part_(f1)
    partB = part_(f2)
    address = analyze(partA, partB, story, f1, f2, p)
    return render_template("compared.html")




if __name__ == '__main__':
    app.run()

