# -*- coding: utf-8 -*-
import os, sys, tqdm, random, re
from tqdm import trange
from bs4 import BeautifulSoup


data_root = '/data/disk1/private/wangmuzi/'
tocfl_write = 'data/detection/hsk/'

tocfl_src = "raw_data/nlptea16cged/Training/CGED16_HSK_TrainingSet.txt"

# lang8_rm_partern = '(.*?)|[.*?f-red]|[.*?f-blue]|[sline].*?[\/sline]'
# lang8_rm_partern = r'\[sline\](.*?)\[\\/sline\]'
lang8_rm_partern = r'\[sline\](.*?)\[\\/sline\]|\[.*?\]|\（.*?\）'
reg = re.compile('<[^>]*>')

def give_hsk_tag(src_dir, write_dir):
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    train_dir = write_dir + 'train.txt'
    train_trg_dir = write_dir + 'train.tag'
    valid_dir = write_dir + 'valid.txt'
    valid_trg_dir = write_dir + 'valid.tag'
    valid_err = write_dir + 'valid.err'
    with open(train_dir, 'w+') as f_train, \
            open(train_trg_dir, 'w+') as f_train_tag, \
            open(valid_dir, 'w+') as f_valid, \
            open(valid_trg_dir, 'w+') as f_valid_tag, \
            open(valid_err, 'w+') as f_err:
        with open(src_dir, 'r',  encoding='UTF-8') as f:
            datas = []
            doc_num = f.readlines()
            cnt = 0
            wrong = 0
            single_data=''
            for line in tqdm.tqdm(doc_num, desc="giving {0} tag".format(src_dir.split('/')[-1])):
                cnt += 1
                if line.strip():
                    single_data += line
                    continue
                datas.append(single_data)
                single_data = ''
            valid_index = random.sample(range(len(datas)), int(0.1 * len(datas)))
            for i in trange(len(datas)):
                soup = BeautifulSoup(datas[i], "html.parser")
                # print(len(soup.find_all("essay")))
                # print(datas[i])
                essay = soup.find('doc')
                try:
                    id = essay.find("text")['id']
                except AttributeError:
                    wrong += 1
                    continue
                txt = essay.find("text").text.strip().replace('\n','')
                tags = len(txt)*["O"]
                sent_err = []
                mistakes = essay.find_all("error")
                for mistake in mistakes:
                    try:
                        type = mistake['type']
                        start = int(mistake['start_off']) - 1
                        end = int(mistake['end_off']) - 1
                        sent_err.append(mistake['start_off']+' '+mistake['end_off']+' '+type)
                    except ValueError:
                        print(cnt)
                    try:
                        if start == end:
                            tags[start] = type + 'b'
                        else:
                            tags[start] = type + 'b'
                            for i in range(start+1, end+1):
                                tags[i] = type + 'i'
                    except IndexError:
                        print('index error:   essay id:{0}, start_off:{1}, end_off:{2}, length：{3}'.format(id, start+1, end+1, len(txt)))
                        continue
                assert len(txt) == len(tags)
                if i not in valid_index:
                    f_train.write(' '.join(txt) + '\n')
                    f_train_tag.write(str(' '.join(tag for tag in tags)) + '\n')
                else:
                    if sent_err == ['']:
                        continue
                    f_valid.write(' '.join(txt) + '\n')
                    f_valid_tag.write(str(' '.join(tag for tag in tags)) + '\n')
                    f_err.write('@'.join(err for err in sent_err) + '\n')
            f.close()
            print(wrong)


def test(trian_dir, valid_dir):
    with open(trian_dir, 'r') as f_train, open(valid_dir, 'r') as f_valid:
        train_em = set()
        valid_em = set()
        line_num = 0
        for l_tr in f_train:
            if len(l_tr) is 0:
                train_em.append(line_num)
        line_num = 0
        for l_vld in f_valid:
            if len(l_vld) is 0:
                valid_em.append(line_num)
        print(train_em, valid_em)


if __name__ == "__main__":
    give_hsk_tag(data_root + tocfl_src, data_root + tocfl_write)
    # test('data/lang8/lang8_train.src.txt', 'data/lang8/lang8_train.trg.txt')
