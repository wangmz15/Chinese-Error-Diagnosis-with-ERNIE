import csv
import tqdm
import thulac
from bs4 import BeautifulSoup

HSK_TEST = '/data/disk1/private/wangmuzi/raw_data/nlptea16cged/Test'
HSK_DATA='/data/disk1/private/wangmuzi/data/detection/hsk'
TOCFL_DATA='/data/disk1/private/wangmuzi/data/detection/tocfl'
ERNIE_DATA='/data/disk1/private/wangmuzi/data/ERNIE/cged_seg'

def get_train_dev():
    with open(HSK_DATA + '/train.txt') as train_txt, \
            open(HSK_DATA + '/train.tag') as train_tag, \
            open(HSK_DATA + '/valid.txt') as valid_txt, \
            open(HSK_DATA + '/valid.tag') as valid_tag, \
            open(ERNIE_DATA + '/train.tsv', 'w') as train, \
            open(ERNIE_DATA + '/valid.tsv', 'w') as valid:
        csvData = [['text_a', 'label']]
        for txt, tag in zip(train_txt, train_tag):
            csvData.append([u"".join(txt.strip().split()), u"".join(tag.strip().split())])
        train_w = csv.writer(train, delimiter="\t")
        train_w.writerows(csvData)

        csvData = [['text_a', 'label']]
        for txt, tag in zip(valid_txt, valid_tag):
            csvData.append([u"".join(txt.strip().split()), u"".join(tag.strip().split())])
        valid_w = csv.writer(valid, delimiter="\t")
        valid_w.writerows(csvData)



def get_test():
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

        # exit(0)
        print(cnt)

        csvData = [['text_a', 'label']]
        for sid, (txt, tag) in test_data.items():
            csvData.append([u"".join(txt), u"".join(tag)])
        train_w = csv.writer(test, delimiter="\t")
        train_w.writerows(csvData)






def get_seg(txt, tag):
    thu1 = thulac.thulac(T2S=True, seg_only=True)  # 默认模式
    csvData = [['text_a', 'label']]
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
        csvData.append([u"".join(txt_seg), u"".join(tag_seg)])
    return csvData





def get_train_dev_seg(HSK_DATA):
    with open(HSK_DATA + '/train.txt') as train_txt, \
            open(HSK_DATA + '/train.tag') as train_tag, \
            open(HSK_DATA + '/valid.txt') as valid_txt, \
            open(HSK_DATA + '/valid.tag') as valid_tag, \
            open(TOCFL_DATA + '/train.txt') as train_txt, \
            open(TOCFL_DATA + '/train.tag') as train_tag, \
            open(TOCFL_DATA + '/valid.txt') as valid_txt, \
            open(TOCFL_DATA + '/valid.tag') as valid_tag, \
            open(ERNIE_DATA + '/train.tsv', 'w') as train, \
            open(ERNIE_DATA + '/valid.tsv', 'w') as valid:
        train_txt = [i.strip().replace('\n','').replace(' ','') for i in train_txt.readlines()]
        train_tag = [i.strip().replace('\n','').split() for i in train_tag.readlines()]
        valid_txt = [i.strip().replace('\n','').replace(' ','') for i in valid_txt.readlines()]
        valid_tag = [i.strip().replace('\n','').split() for i in valid_tag.readlines()]

        train_w = csv.writer(train, delimiter="\t")
        train_w.writerows(get_seg(train_txt,train_tag))
        valid_w = csv.writer(valid, delimiter="\t")
        valid_w.writerows(get_seg(valid_txt,valid_tag))

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
        txts, tags = [], []
        for id, (txt, tag) in test_data.items():
            txts.append(''.join(txt))
            tags.append(tag)
        train_w.writerows(get_seg(txts, tags))

if __name__ == "__main__":
    # get_test()
    # get_train_dev()
    # get_train_dev_seg()
    get_test_seg()