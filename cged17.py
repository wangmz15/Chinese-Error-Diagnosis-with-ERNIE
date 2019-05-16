import csv
import tqdm, random, re
from tqdm import trange
import thulac
from bs4 import BeautifulSoup
from collections import namedtuple


TEST = '/data/disk1/private/wangmuzi/raw_data/cged17'
TRAIN_RAW = '/data/disk1/private/wangmuzi/raw_data/cged17'
HSK_DATA='/data/disk1/private/wangmuzi/data/detection/hsk'
TOCFL_DATA='/data/disk1/private/wangmuzi/data/detection/tocfl'
ERNIE_DATA='/data/disk1/private/wangmuzi/data/ERNIE/cged17'

def get_train_dev(src_dir):
    csvData = [['text_a', 'label']]
    with open(src_dir, 'r', encoding='UTF-8') as f, open(ERNIE_DATA + '/train.tsv', 'w') as train:
        datas = []
        cnt = 0
        lines = ''.join(f.readlines())
        soup = BeautifulSoup(lines, "html.parser")
        datas = soup.find_all('doc')
        wrong = 0
        print(len(datas))
        for essay in datas:
            if essay == None:
                continue
            try:
                id = essay.find("text")['id']
            except:
                wrong += 1
                continue
            txt = essay.find("text").text.strip().replace('\n', '')
            tags = len(txt) * ["O"]
            sent_err = []
            mistakes = essay.find_all("error")
            for mistake in mistakes:
                try:
                    type = mistake['type']
                    start = int(mistake['start_off']) - 1
                    end = int(mistake['end_off']) - 1
                    sent_err.append(mistake['start_off'] + ' ' + mistake['end_off'] + ' ' + type)
                except ValueError:
                    print(cnt)
                try:
                    if start == end:
                        tags[start] = type + 'b'
                    else:
                        tags[start] = type + 'b'
                        for i in range(start + 1, end + 1):
                            tags[i] = type + 'i'
                except IndexError:
                    print('index error:   essay id:{0}, start_off:{1}, end_off:{2}, length：{3}'.format(id, start + 1,
                                                                                                       end + 1,
                                                                                                       len(txt)))
                    continue
            assert len(txt) == len(tags)
            csvData.append([u"".join(txt), u"".join(tags)])
        train_w = csv.writer(train, delimiter="\t")
        train_w.writerows(csvData)
        print('wrong number:', wrong)



def get_test():
    with open(TEST + '/test.HSK.Input.txt') as test_txt, \
            open(TEST + '/test.truth.txt') as test_tag, \
            open(ERNIE_DATA + '/test.tsv', 'w') as test:
        test_data = {}
        cnt = 0
        for line in tqdm.tqdm(test_txt.readlines()):
            sid, content = line.split(')')
            sid = sid[5:]
            content = [i for i in content.strip().rstrip()]
            # print(sid,content)
            # break
            test_data[sid] = (content,[])
        for line in tqdm.tqdm(test_tag.readlines()):
            sid = line.split(',')[0]
            content = test_data[sid][0]
            tag = ['O'] * len(content)
            if len(test_data[sid][1]):
                # print(test_data[sid][1])
                tag = test_data[sid][1]
                # print(content, tag)
            if len(line.split(',')) != 2:
                start = int(line.split(',')[1].strip())-1
                end = int(line.split(',')[2].strip())-1
                type = line.split(',')[3].strip().rstrip()
                try:
                    tag[start] = type + 'b'
                    for i in range(start + 1, end + 1):
                        tag[i] = type + 'i'
                except:
                    cnt += 1
                    print(sid, content)
            # print(content, tag)
            # break
            test_data[sid] = (content, tag)

        # exit(0)
        print(cnt)

        csvData = [['text_a', 'label']]
        for sid, (txt, tag) in test_data.items():
            csvData.append([u"".join(txt), u"".join(tag)])
        train_w = csv.writer(test, delimiter="\t")
        train_w.writerows(csvData)


def cut_sent(para, tag):
    para = re.sub('([。\t！？；\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    len_p = len(''.join(para.split("\n")))
    start = 0
    if len_p == len(tag):
        tags = []
        for p in para.split('\n'):
            tags.append(tag[start:start+len(p)])
            start += len(p)
        return para.split("\n"), tags
    else:
        return [para], [tag]

global_csvData =  []

def get_seg(txt, tag):
    thu1 = thulac.thulac(T2S=True, seg_only=True)  # 默认模式
    # csvData = [['text_a', 'label']]
    for txt, tag in zip(txt, tag):
        txt_seg = [word[0] for word in thu1.cut(txt, text=False)]
        tag_seg = ["O"] * len(txt_seg)
        tag_start = 0
        for j in range(len(txt_seg)):
            tmp_tag = tag[tag_start: tag_start + len(txt_seg[j])]
            for i in range(len(tmp_tag)):
                if tmp_tag[i] == 'O':
                    continue
                if j == 0:
                    tag_seg[j] = tmp_tag[i][0] + 'b'
                else:
                    if tag_seg[j - 1] == 'O' or tag_seg[j - 1][0] != tmp_tag[i][0]:
                        tag_seg[j] = tmp_tag[i][0] + 'b'
                    else:  # tag_seg[j-1][0] == tmp_tag[i][0]
                        tag_seg[j] = tmp_tag[i][0] + 'i'
            tag_start += len(txt_seg[j])
        assert len(txt_seg) == len(tag_seg)
        global_csvData.append([u"".join(txt_seg), u"".join(tag_seg)])





def get_train_dev_seg():
    with open(ERNIE_DATA + '/train.tsv') as fr, open(ERNIE_DATA + '/train62.tsv', 'w') as fw:
        reader = csv.reader(fr, delimiter="\t", quotechar=None)
        writer = csv.writer(fw, delimiter="\t")
        writer.writerow(['text_a', 'label'])

        final_txt = []
        final_tag = []
        for item in tqdm.tqdm(reader):
            if reader.line_num == 1:
                continue
            param = ''.join(item[0].split(u""))
            tags = item[1].split(u"")
            a, b = cut_sent(param, tags)
            # print(a,b)
            final_txt.extend(a)
            final_tag.extend(b)
        # get_seg(final_txt,final_tag)
        final_txt, final_tag = cut_short(final_txt, final_tag)
        get_seg(final_txt, final_tag)
        for i in range(len(global_csvData)):
            writer.writerow(global_csvData[i])


def get_test_seg():
    with open(HSK_TEST + '/CGED16_HSK_Test_Input.txt') as test_txt, \
            open(HSK_TEST + '/CGED16_HSK_Test_Truth.txt') as test_tag, \
            open(ERNIE_DATA + '/test.tsv', 'w') as test:
        test_data = {}
        cnt = 0
        for line in tqdm.tqdm(test_txt.readlines()):
            sid, content = line.split(')')
            sid = sid[5:]
            content = [i for i in content.strip().rstrip()]
            # print(sid,content)
            # break
            test_data[sid] = (content,[])
        for line in tqdm.tqdm(test_tag.readlines()):
            sid = line.split(',')[0]
            content = test_data[sid][0]
            tag = ['O'] * len(content)
            if len(test_data[sid][1]):
                # print(test_data[sid][1])
                tag = test_data[sid][1]
                # print(content, tag)
            if len(line.split(',')) != 2:
                start = int(line.split(',')[1].strip())-1
                end = int(line.split(',')[2].strip())-1
                type = line.split(',')[3].strip().rstrip()
                try:
                    tag[start] = type + 'b'
                    for i in range(start + 1, end + 1):
                        tag[i] = type + 'i'
                except:
                    cnt += 1
                    print(sid, content)
            # print(content, tag)
            # break
            test_data[sid] = (content, tag)


        train_w = csv.writer(test, delimiter="\t")
        train_w.writerow(['text_a', 'label'])
        txts, tags = [], []
        for id, (txt, tag) in test_data.items():
            txts.append(''.join(txt))
            tags.append(tag)

        train_w.writerows(get_seg(txts, tags))

def cut_sent_short(para, tag):
    para = re.sub('([。\t！？；\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    len_p = len(''.join(para.split("\n")))
    start = 0
    if len_p == len(tag):
        tags = []
        for p in para.split('\n'):
            tags.append(tag[start:start+len(p)])
            start += len(p)
        return para.split("\n"), tags
    else:
        return [para], [tag]

LEN = 62
def cut_short(final_txt, final_tag):
    new_txt, new_tag = [], []
    split_list = ['，',',','。']
    for txt, tag in zip(final_txt,final_tag):
        if len(tag) <= LEN:
            if len(tag) <= 5:
                continue
            new_txt.append(txt)
            new_tag.append(tag)
        else:
            # pos = min(LEN*2-1, len(tag)-1)
            pos = min(LEN-1,len(tag)-1)
            start = 0
            try:
                while pos >= start:
                    if txt[pos] in split_list:
                        if pos+1-start > 5:
                            new_txt.append(txt[start: pos+1])
                            new_tag.append(tag[start: pos+1])
                            print(txt)
                            print(start,pos,len(tag))
                            print(txt[start: pos+1])
                        start = pos+1
                        pos = min(pos+LEN-1, len(tag)-1)
                        # print(start, pos, len(tag))
                        if pos == len(tag) -1:
                            break
                    pos -= 1
                new_txt.append(txt[start: pos+1])
                new_tag.append(tag[start: pos+1])
                print(txt)
                print(start, pos, len(tag))
                print(txt[start: pos + 1])
            except:
                print(txt)
                print(start,pos)
    return new_txt, new_tag




def get_cut_sent_test():
    with open(ERNIE_DATA + '/test.tsv') as fr, open(ERNIE_DATA + '/test62.tsv', 'w') as fw:
        reader = csv.reader(fr, delimiter="\t", quotechar=None)
        writer = csv.writer(fw, delimiter="\t")
        writer.writerow(['text_a', 'label'])

        final_txt = []
        final_tag = []
        for item in tqdm.tqdm(reader):
            if reader.line_num == 1:
                continue
            param = ''.join(item[0].split(u""))
            tags = item[1].split(u"")
            a, b = cut_sent(param,tags)
            # print(a,b)
            final_txt.extend(a)
            final_tag.extend(b)
        # get_seg(final_txt,final_tag)
        final_txt, final_tag = cut_short(final_txt,final_tag)
        for txt,tag in zip(final_txt, final_tag):
            writer.writerow([u"".join([i for i in txt]), u"".join(tag)])
        # exit(0)



if __name__ == "__main__":
    # get_test()
    # get_train_dev(TRAIN_RAW+'/train.release.xml')
    get_train_dev_seg()
    # get_test_seg()
    # get_cut_sent_test()