import csv
import tqdm, random, re
from tqdm import trange
import thulac
from bs4 import BeautifulSoup
from collections import namedtuple

ERNIE_DATA='/data/disk1/private/wangmuzi/data/ERNIE/cged'





if __name__ == "__main__":
    train_lines = []
    train62_lines = []
    test_lines = []
    test62_lines = []
    title = ''
    for i , post in enumerate(['15', '_seg', '17', '18']):
        title = open(ERNIE_DATA+post+'/train.tsv').readline()
        train_lines.extend(open(ERNIE_DATA+post+'/train.tsv').readlines()[1:])
        train62_lines.extend(open(ERNIE_DATA+post+'/train62.tsv').readlines()[1:])
        try:
            test_lines.extend(open(ERNIE_DATA + post + '/test.tsv').readlines()[1:])
            test62_lines.extend(open(ERNIE_DATA + post + '/test62.tsv').readlines()[1:])
        except:
            print(post)

    random.shuffle(train_lines)
    random.shuffle(train62_lines)
    train_lines.insert(0, title)
    train62_lines.insert(0, title)

    random.shuffle(test_lines)
    random.shuffle(test62_lines)
    test_lines.insert(0, title)
    test62_lines.insert(0, title)

    with open(ERNIE_DATA+'_all/train.tsv', 'w') as train,open(ERNIE_DATA+'_all/train62.tsv', 'w') as train62 :
        train.writelines(train_lines)
        train62.writelines(train62_lines)
