#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse,csv
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model.classifier_remodel import *

from model.ernie import ErnieModel

def create_model(args,
                 pyreader_name,
                 ernie_config,
                 is_prediction=False):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        # shapes=[[args.batch_size, args.max_seq_len, 1], [args.batch_size, args.max_seq_len, 1],
        #         [args.batch_size, args.max_seq_len, 1], [args.batch_size, args.max_seq_len, 1],
        #         [args.batch_size, args.max_seq_len, 1], [args.batch_size, 1]],
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        use_double_buffer=True,
        name=pyreader_name)

    (src_ids, sent_ids, pos_ids, self_attn_mask, labels,
     seq_lens) = fluid.layers.read_file(pyreader)
    # print('src_ids:', src_ids.shape)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        self_attn_mask=self_attn_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    enc_out = ernie.get_sequence_output()
    # enc_out_tensor = np.array(fluid.global_scope().find_var(
    #     enc_out.name).get_tensor())

    ################### classifier_concatenate_left_right ######################
    # words = [layers.create_tensor(dtype='float32')] * enc_out.shape[1]
    # for i in range(enc_out.shape[1]):
    #    j = fluid.layers.fill_constant(shape=[1], dtype='int64', value=i)
    #    words[i] = layers.array_read(enc_out,j)
    # enc_out_table = fluid.layers.control_flow.lod_rank_table(enc_out)
    # enc_out_array = layers.control_flow.lod_tensor_to_array(x=enc_out, table=enc_out_table)
    # counter = fluid.layers.zeros(shape=[1], dtype='int64')
    # array_len = fluid.layers.fill_constant(shape=[1], dtype='int64', value=enc_out.shape[1])
    # cond = fluid.layers.less_than(x=counter, y=array_len)
    # while_op = fluid.layers.While(cond=cond)
    # for j in range(64):
    #    words[j] = fluid.layers.array_read(array=enc_out_array, i=counter)
    #    fluid.layers.increment(x=counter, value=1, in_place=True)
    # fluid.layers.less_than(x=counter, y=array_len, cond=cond)
    # j += 1
    #   print(words[j].shape)

    # new_enc_out = enc_out
    # new_enc_out = classifier_concat_left_middle_right(enc_out)
    # new_enc_out = classifier_concat_left2_middle_right2(enc_out)
    # new_enc_out = classifier_maxPool_left_middle_right_33(enc_out)
    # new_enc_out = classifier_maxPool_left_middle_right_33(enc_out)
    # new_enc_out = classifier_maxPool_left_right_21_concat_middle(enc_out)
    # new_enc_out = classifier_avgPool_left_right_21_concat_middle(enc_out)
    # new_enc_out = classifier_maxPool_left_middle_right_31(enc_out)
    # new_enc_out = classifier_avgPool_left_middle_right_31(enc_out)
    # new_enc_out = classifier_maxPool_left_middle_right_31_concat_middle(enc_out)
    # new_enc_out = classifier_avgPool_left_middle_right_31_concat_middle(enc_out)

    # new_enc_out = classifier_windowAdd_left_right_concat_middle(enc_out) #1 but version changed 0
    new_enc_out = classifier_windowAdd_left2_right2_concat_middle(enc_out) #1



    # new_enc_out, attn_scores, max_js = classifier_maxAttn1_concat_middle(enc_out,self_attn_mask) #1 yesterday
    # new_enc_out, attn_scores, max_js = classifier_maxAttnLeft1Right1_concat_middle(enc_out,self_attn_mask) #1 yesterday

    # new_enc_out, attn_scores, max_js = classifier_weightedAdd_all_attention(enc_out,self_attn_mask)
    # new_enc_out, attn_scores, max_js = classifier_weightedAdd_all_attention_concat_middle(enc_out, self_attn_mask) #1 micro new

    logits = fluid.layers.fc(
        input=new_enc_out,
        size=args.num_labels,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_out_b", initializer=fluid.initializer.Constant(0.)))
    print('logits shape:',logits.shape)

    print('labels shape:', labels.shape)
    ret_labels = fluid.layers.reshape(x=labels, shape=[-1,1])
    ret_infers = fluid.layers.reshape(x=fluid.layers.argmax(logits, axis=2), shape=[-1,1])
    print('ret_labels shape:',ret_labels.shape)
    print('ret_infers shape:',ret_infers.shape)
    labels = fluid.layers.flatten(labels, axis=2)
    print('labels shape:',labels.shape)

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=fluid.layers.flatten(logits, axis=2),
        label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    graph_vars = {"loss": loss,
                  "probs": probs,
                  "labels": ret_labels,
                  "infers": ret_infers,
                  "seq_lens": seq_lens,
                  # "attn_scores":attn_scores,
                  # "max_js": max_js,
                  "enc_out": new_enc_out,

                  }


    for k, v in graph_vars.items():
        v.persistable=True

    return pyreader, graph_vars

def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):

    def extract_bio_chunk(seq):
        chunks = []
        cur_chunk = None
        null_index = tag_num - 1
        for index in range(len(seq)):
            tag = seq[index]
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st":index, "en": index + 1, "type": tag_type}

            else:
                if cur_chunk is None:
                    cur_chunk = {"st":index, "en": index + 1, "type": tag_type}
                    continue

                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"]  = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {"st":index, "en": index + 1, "type": tag_type}

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    null_index = tag_num - 1
    num_label = 0
    num_infer = 0
    num_correct = 0
    labels = np_labels.reshape([-1]).astype(np.int32).tolist()
    infers = np_infers.reshape([-1]).astype(np.int32).tolist()
    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    for dev_index in range(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in range(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            infer_chunks = extract_bio_chunk(infers[seq_st:seq_en])
            label_chunks = extract_bio_chunk(labels[seq_st:seq_en])
            num_infer += len(infer_chunks)
            num_label += len(label_chunks)

            infer_index = 0
            label_index = 0
            while label_index < len(label_chunks) and infer_index < len(infer_chunks):
                if infer_chunks[infer_index]["st"] < label_chunks[label_index]["st"]:
                    infer_index += 1
                elif infer_chunks[infer_index]["st"] > label_chunks[label_index]["st"]:
                    label_index += 1
                else:
                    if infer_chunks[infer_index]["en"] == label_chunks[label_index]["en"] and \
                            infer_chunks[infer_index]["type"] == label_chunks[label_index]["type"]:
                        num_correct += 1

                    infer_index += 1
                    label_index += 1

        base_index += max_len * len(lens)

    return num_label, num_infer, num_correct

def calculate_f1(num_label, num_infer, num_correct):
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def evaluate(exe, program, pyreader, graph_vars, tag_num, eval_phase, test_text = '', test_model_name='', dev_count=1):
    fetch_list = [graph_vars["labels"].name,
                  graph_vars["infers"].name,
                  graph_vars["seq_lens"].name]
    # print(fetch_list)
    if eval_phase == "train":
        fetch_list.append(graph_vars["loss"].name)
        if "learning_rate" in graph_vars:
            fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=fetch_list)
        np_labels, np_infers, np_lens, np_loss  = outputs[:4]
        num_label, num_infer, num_correct = chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count)
        precision, recall, f1 = calculate_f1(num_label, num_infer, num_correct)
        res = {"precision": precision, "recall": recall, "f1": f1, "loss": np.mean(np_loss)}
        if "learning_rate" in graph_vars:
            res["lr"] = float(outputs[4][0])
            # outputs['lr'] = graph_vars['learning_rate']
        return res
    elif eval_phase == "dev":
        total_label, total_infer, total_correct = 0.0, 0.0, 0.0
        time_begin = time.time()
        pyreader.start()
        while True:
            try:
                np_labels, np_infers, np_lens = exe.run(program=program, fetch_list=fetch_list)
                label_num, infer_num, correct_num = chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count)
                total_infer += infer_num
                total_label += label_num
                total_correct += correct_num

            except fluid.core.EOFException:
                pyreader.reset()
                break

        precision, recall, f1 = calculate_f1(total_label, total_infer, total_correct)
        time_end = time.time()

        print("[%s evaluation] f1: %f, precision: %f, recall: %f, elapsed time: %f s" %
              (eval_phase, f1, precision, recall, time_end - time_begin))
    else:
        total_label1, total_infer1, total_correct1, total_acc_correct1, total_example1 = 0.0, 0.0, 0.0, 0.0, 0.0
        total_label2, total_infer2, total_correct2, total_acc_correct2, total_example2 = 0.0, 0.0, 0.0, 0.0, 0.0
        total_label3, total_infer3, total_correct3, total_acc_correct3, total_example3 = 0.0, 0.0, 0.0, 0.0, 0.0

        time_begin = time.time()
        pyreader.start()
        to_write = []
        index = 1

        labels_map = {0:'Rb',1:'Ri',2:'Sb',3:'Si',4:'Mb',5:'Mi',6:'Wb',7:'Wi',8:'O'}
        while True:
            try:
                np_labels, np_infers, np_lens = exe.run(program=program, fetch_list=fetch_list)
                np_tmp = np_labels
                np_labels = np_infers
                np_infers = np_tmp
                # print(len(np_labels), len(np_infers), len(np_lens))

                predicts_id = []
                origins_id = []
                len_of_sent = int(len(np_labels)/len(np_lens))
                for j in range(len(np_lens)):
                #     print(j, range(j*len_of_sent+1, (j+1)*len_of_sent-1)[0], range(j*len_of_sent+1, (j+1)*len_of_sent-1)[-1])
                    predicts_id.append([np_labels[i][0] for i in range(j*len_of_sent+1, (j+1)*len_of_sent-1)])
                    origins_id.append([np_infers[i][0] for i in range(j*len_of_sent+1, (j+1)*len_of_sent-1)])

                for predict,origin in zip(predicts_id,origins_id):
                    # print('detection level')
                    total_example1 += 1
                    jiancechucuo = False
                    zhendeyoucuo = False
                    jianceduile = False
                    duidejianceduile = True
                    for i in range(len(predict)):
                        if predict[i] != 8:
                            jiancechucuo = True
                            duidejianceduile = False
                        if origin[i] != 8:
                            zhendeyoucuo = True
                            duidejianceduile = False
                        if jiancechucuo and zhendeyoucuo:
                            jianceduile = True
                            break
                    if jianceduile:
                        total_correct1 += 1
                    if jiancechucuo:
                        total_label1 += 1
                    if zhendeyoucuo:
                        total_infer1 += 1
                    if duidejianceduile:
                        total_acc_correct1 += 1

                    # print('ident level')
                    err_types_find = set()
                    err_types_origin = set()
                    for o,p in zip(origin, predict):
                        if p != 8 and p%2 == 0:
                            err_types_find.add(np_labels[i][0])
                        if o != 8 and o%2 == 0:
                            err_types_origin.add(np_labels[i][0])
                    total_label2 += len(err_types_find)
                    total_infer2 += len(err_types_origin)
                    total_correct2 += len(err_types_origin & err_types_find)
                    if len(err_types_origin) != 0:
                        total_example2 += len(err_types_origin)
                    else:
                        total_example2 += 1
                    if total_infer2 == 0 and total_label3 == 0:
                        total_acc_correct2 += 1

                    # print('position level')
                    err_pos_find = list()
                    err_pos_origin = list()
                    i = 0
                    while i < len(predict):
                        if predict[i] != 8 and predict[i]%2 == 0:
                            type = predict[i]
                            start = i
                            end = start
                            while end < len(predict)-1 and predict[end+1] == type+1:
                                end += 1
                            err_pos_find.append([start, end, type])
                            i = end+1
                        else:
                            i += 1
                    i = 0
                    while i < len(origin):
                        if origin[i] != 8 and origin[i] % 2 == 0:
                            type = origin[i]
                            start = i
                            end = start
                            while end < len(origin)-1 and origin[end + 1] == type + 1:
                                end += 1
                            err_pos_origin.append([start, end, type])
                            i = end + 1
                        else:
                            i += 1
                    total_label3 += len(err_pos_find)
                    total_infer3 += len(err_pos_origin)
                    if total_infer3 == 0 and total_label3 == 0:
                        total_acc_correct3 += 1
                    if len(err_pos_origin) != 0:
                        total_example3 += len(err_pos_origin)
                    else:
                        total_example3 += 1
                    flag=False
                    for i in err_pos_find:
                        if i in err_pos_origin:
                            flag = True
                            total_correct3 += 1
                    if 'final' in eval_phase and test_model_name:
                        res = []
                        # if flag:
                        #     res.append('correct:')
                        # else:
                        #     res.append('wrong:')
                        for test_pos,[p,o] in enumerate(zip(predict, origin)):
                            res.append(labels_map[o]+labels_map[p]+' ')
                        # print(len(res))
                        to_write.append(res)
                    # print(len(to_write), index)
                    index += 1
            except fluid.core.EOFException:
                pyreader.reset()
                break
        # print(len(to_write), )
        if 'final' in eval_phase and test_model_name:
            test_result_write_dir = '/data/disk1/private/wangmuzi/data/ERNIE/cged_seg/test_result/'+ test_model_name.split('/')[-2]
            if not os.path.exists(test_result_write_dir):
                os.makedirs(test_result_write_dir)
            fw = open(test_result_write_dir+'/_'+test_model_name.split('/')[-1]+'_'+test_text.split('/')[-1].split('.')[0]+'.txt', 'w')
            test_text =  open(test_text).readlines()[1:]
            for i, [test_text,lin] in enumerate(zip(test_text,to_write)):
                # fw.write(lin)
                test_text = test_text.split('\t')[0].split(u"")
                # print(test_text)
                fw.write(' '.join([char+op for char, op in zip(test_text, lin)]) + '\n')

        precision1, recall1, f11 = calculate_f1(total_label1, total_infer1, total_correct1)
        acc1 = (total_correct1 + total_acc_correct1) * 1.0/total_example1
        precision2, recall2, f12 = calculate_f1(total_label2, total_infer2, total_correct2)
        acc2 = (total_correct2 + total_acc_correct2) * 1.0/total_example2
        precision3, recall3, f13 = calculate_f1(total_label3, total_infer3, total_correct3)
        acc3 = (total_correct3 + total_acc_correct3) * 1.0/total_example3
        time_end = time.time()

        print("[%s evaluation] acc: %f, f1: %f, precision: %f, recall: %f, elapsed time: %f s" %
              (eval_phase, acc1, f11, precision1, recall1, time_end - time_begin))
        print("[%s evaluation] acc: %f, f1: %f, precision: %f, recall: %f, elapsed time: %f s" %
              (eval_phase, acc2, f12, precision2, recall2, time_end - time_begin))
        print("[%s evaluation] acc: %f, f1: %f, precision: %f, recall: %f, elapsed time: %f s" %
              (eval_phase, acc3, f13, precision3, recall3, time_end - time_begin))
        outputs = {"f1": f13}
        return outputs