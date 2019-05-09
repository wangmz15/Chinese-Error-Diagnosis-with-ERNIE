import paddle, sys
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model.transformer_encoder import *

def classifier_concat_left_middle_right(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    new_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        new_words[i] = layers.concat([ words[max(0,i-1)], words[i], words[min(l-1,i+1)]],axis=2)
    new_enc_out = layers.concat(new_words,axis=1)
    return new_enc_out

def classifier_concat_left2_middle_right2(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    new_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        new_words[i] = layers.concat([words[max(0,i-2)], words[max(0,i-1)], words[i], words[min(l-1,i+1)], words[min(l-1,i+2)]],axis=2)
    new_enc_out = layers.concat(new_words,axis=1)
    return new_enc_out

def classifier_maxPool_left_middle_right_33(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[i], words[min(l-1,i+1)]],axis=1), [1])
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=3,
                  pool_type='max',
                  pool_stride=1,
                  # pool_padding=1,
                  global_pooling=False)
        pool_words[i] = layers.squeeze(pool_words[i], [1])

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out

def classifier_avgPool_left_middle_right_33(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[i], words[min(l-1,i+1)]],axis=1), [1])
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=3,
                  pool_type='avg',
                  pool_stride=1,
                  # pool_padding=1,
                  global_pooling=False)
        pool_words[i] = layers.squeeze(pool_words[i], [1])

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out

def classifier_maxPool_left_right_21_concat_middle(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[min(l-1,i+1)]],axis=1), [1])
        # print(pool_words[i].shape)
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=[2,1],
                  pool_type='max',
                  pool_stride=1,
                  pool_padding=[0,0],
                  global_pooling=False)

        pool_words[i] = layers.squeeze(pool_words[i], [1])
        pool_words[i] = layers.concat([words[i], pool_words[i]], axis=2)

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out

def classifier_avgPool_left_right_21_concat_middle(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[min(l-1,i+1)]],axis=1), [1])
        # print(pool_words[i].shape)
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=[2,1],
                  pool_type='avg',
                  pool_stride=1,
                  pool_padding=[0,0],
                  global_pooling=False)

        pool_words[i] = layers.squeeze(pool_words[i], [1])
        pool_words[i] = layers.concat([words[i], pool_words[i]], axis=2)

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out

def classifier_maxPool_left_middle_right_31(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[i], words[min(l-1,i+1)]],axis=1), [1])
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=[3,1],
                  pool_type='max',
                  pool_stride=1,
                  # pool_padding=1,
                  global_pooling=False)
        pool_words[i] = layers.squeeze(pool_words[i], [1])

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out

def classifier_avgPool_left_middle_right_31(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[i], words[min(l-1,i+1)]],axis=1), [1])
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=[3,1],
                  pool_type='avg',
                  pool_stride=1,
                  # pool_padding=1,
                  global_pooling=False)
        pool_words[i] = layers.squeeze(pool_words[i], [1])

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out



def classifier_maxPool_left_middle_right_31_concat_middle(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[i], words[min(l-1,i+1)]],axis=1), [1])
        # print(pool_words[i].shape)
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=[3,1],
                  pool_type='max',
                  pool_stride=1,
                  pool_padding=[0,0],
                  global_pooling=False)

        pool_words[i] = layers.squeeze(pool_words[i], [1])
        pool_words[i] = layers.concat([words[i], pool_words[i]], axis=2)

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out


def classifier_avgPool_left_middle_right_31_concat_middle(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    pool_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        pool_words[i] = layers.unsqueeze(layers.concat([words[max(0,i-1)], words[i], words[min(l-1,i+1)]],axis=1), [1])
        # print(pool_words[i].shape)
        pool_words[i] = layers.pool2d(
                  input=pool_words[i],
                  pool_size=[3,1],
                  pool_type='avg',
                  pool_stride=1,
                  pool_padding=[0,0],
                  global_pooling=False)

        pool_words[i] = layers.squeeze(pool_words[i], [1])
        pool_words[i] = layers.concat([words[i], pool_words[i]], axis=2)

    new_enc_out = layers.concat(pool_words,axis=1)
    return new_enc_out

def classifier_windowAdd_left_right_concat_middle(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    new_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        to_concat = 0.5*(0.5*words[max(0,i-1)] + 0.5*words[min(l-1,i+1)])
        new_words[i] = layers.concat([ words[i], to_concat],axis=2)
    new_enc_out = layers.concat(new_words,axis=1)
    print(new_enc_out)
    return new_enc_out


def classifier_windowAdd_left2_right2_concat_middle(enc_out):
    words = layers.split(enc_out,enc_out.shape[1],1)
    new_words = [layers.create_tensor(dtype='float32')]*enc_out.shape[1]
    l = enc_out.shape[1]
    for i in range(0, l):
        to_concat = (1/3)*(0.5*words[max(0,i-2)] + 0.5*words[min(l-1,i+2)])+\
                    (1/2)*(0.5*words[max(0,i-1)] + 0.5*words[min(l-1,i+1)])
        new_words[i] = layers.concat([ words[i], to_concat],axis=2)
    new_enc_out = layers.concat(new_words,axis=1)
    print(new_enc_out)
    return new_enc_out

def classifier_avgAdd_attention1_middle(enc_out,self_attn_mask):
    n_head_self_attn_mask = fluid.layers.stack(
        x=[self_attn_mask] * 1, axis=1)
    n_head_self_attn_mask.stop_gradient = True
    attn_scores = multi_head_attention(
        pre_process_layer(
            enc_out,
            'da',
            0.0,
            name='classifier_pre_att'),
        None,
        None,
        attn_bias=n_head_self_attn_mask,
        d_key=768,
        d_value=768,
        d_model=768,
        n_head=1,
        dropout_rate=0.0,
        param_initializer=None,
        name= 'classifier_att_score',
        attention_only=True)

    a_sent_scores = layers.split(attn_scores, attn_scores.shape[1], 1)
    a_sent = layers.split(enc_out, enc_out.shape[1], 1)
    new_words = [layers.create_tensor(dtype='float32')] * enc_out.shape[1]
    indices_list = []
    out_list = []


    for j, word_scores in enumerate(a_sent_scores):
        out,ind = layers.argsort(word_scores,axis=2)
        indices_list.append(ind)
        out_list.append(out)
        indices = layers.cast(ind, dtype='float32')
        indices -= (enc_out.shape[1]-3)
        indices = layers.clip(indices,min=0.0, max=0.5)

        new_words[j] = layers.matmul(indices, enc_out)

    new_enc_out = layers.concat(new_words, axis=1)
    print(new_enc_out)

    max_js = layers.concat(indices_list, axis=1)
    return new_enc_out, attn_scores, max_js



def classifier_maxAttn1_concat_middle(enc_out,self_attn_mask):
    # n_head_self_attn_mask = fluid.layers.stack(
    #     x=[self_attn_mask] * 1, axis=1)
    # n_head_self_attn_mask.stop_gradient = True
    # self_attn_mask.stop_gradient = True
    attn_scores = multi_head_attention(
        pre_process_layer(enc_out,'da',0.0,name='classifier_pre_att'),
        None,None,attn_bias=self_attn_mask,
        d_key=768,d_value=768,d_model=768,n_head=1,
        dropout_rate=0.0,param_initializer=None,name= 'classifier_att_score',attention_only=True)

    a_sent_scores = layers.split(attn_scores, attn_scores.shape[1], 1)
    # print(a_sent_scores[31])
    a_sent = layers.split(enc_out, enc_out.shape[1], 1)
    # print(a_sent[31])
    new_words = [layers.create_tensor(dtype='float32')] * enc_out.shape[1]
    indices_list = []
    word_scores_removej_list = []
    for j, [word_scores, word_vec] in enumerate(zip(a_sent_scores, a_sent)):
        word_scores = layers.split(word_scores, word_scores.shape[2], 2)
        # for t in word_scores:
        #     t.stop_gradient=True
        word_scores_removej = word_scores[0:j] + word_scores[j+1:len(word_scores)]
        word_scores_removej = layers.concat(word_scores_removej, axis=2)
        # word_scores_removej = layers.concat([l for i,l in enumerate (word_scores) if i != j], axis=2)
        # word_scores_removej_list.append(word_scores_removej)

        max_j = layers.argmax(word_scores_removej, axis=2)
        indices = layers.one_hot(max_j, word_scores_removej.shape[-1])
        indices_list.append(indices)
        indices = layers.unsqueeze(layers.cast(indices, dtype='float32'),[1])

        enc_out_split = layers.split(enc_out, enc_out.shape[1], 1)
        # for t in enc_out_split:
        #     t.stop_gradient=True
        enc_out_split = enc_out_split[0:j] + enc_out_split[j+1:len(enc_out_split)]
        enc_out_removej = layers.concat(enc_out_split, axis=1)

        # print(enc_out_split[31])
        # print(len(enc_out_split))
        # enc_out_removej = layers.concat([l for p,l in enumerate(layers.split(enc_out, enc_out.shape[1], 1)) if p != j], axis=1)
        # print(indices)
        # print(enc_out_removej)
        new_words[j] = layers.concat([word_vec, layers.matmul(indices, enc_out_removej)], axis=2)
    # attn_scores = layers.concat(word_scores_removej_list, axis=1)
    max_js = layers.concat(indices_list, axis=1)
    new_enc_out = layers.concat(new_words, axis=1)
    # print(new_enc_out)
    return new_enc_out, attn_scores, max_js


def classifier_maxAttnLeft1Right1_concat_middle(enc_out,self_attn_mask):
    # n_head_self_attn_mask = fluid.layers.stack(
    #     x=[self_attn_mask] * 1, axis=1)
    # n_head_self_attn_mask.stop_gradient = True
    attn_scores = multi_head_attention(
        pre_process_layer(enc_out,'da',0.0,name='classifier_pre_att'),
        None,None,attn_bias=self_attn_mask,
        d_key=768,d_value=768,d_model=768,n_head=1,
        dropout_rate=0.0,param_initializer=None,name= 'classifier_att_score',attention_only=True)

    a_sent_scores = layers.split(attn_scores, attn_scores.shape[1], 1)
    a_sent = layers.split(enc_out, enc_out.shape[1], 1)
    new_words = [layers.create_tensor(dtype='float32')] * enc_out.shape[1]
    max_js = layers.fill_constant(shape=[1], dtype='float32', value=0.0)
    indices_list = []
    word_scores_removej_list = []

    for j, [word_scores, word_vec] in enumerate(zip(a_sent_scores, a_sent)):
        if j != 0:
            word_scores_l = layers.concat([l for i,l in enumerate(layers.split(word_scores, word_scores.shape[2], 2)) if i < j], axis=2)
            max_j_l = layers.argmax(word_scores_l, axis=2)
            indices_l = layers.one_hot(max_j_l, word_scores_l.shape[-1])
            indices = layers.unsqueeze(layers.cast(indices_l, dtype='float32'), [1])
            enc_out_l = layers.concat(
                [l for i, l in enumerate(layers.split(enc_out, enc_out.shape[1], 1)) if i < j], axis=1)
            l_attn = layers.matmul(indices, enc_out_l)
        else:
            l_attn = word_vec

        if j != len(a_sent_scores)-1:
            word_scores_r = layers.concat([l for i,l in enumerate(layers.split(word_scores, word_scores.shape[2], 2)) if i > j], axis=2)
            max_j_r = layers.argmax(word_scores_r, axis=2)
            indices_r = layers.one_hot(max_j_r, word_scores_r.shape[-1])
            indices = layers.unsqueeze(layers.cast(indices_r, dtype='float32'), [1])
            enc_out_r = layers.concat(
                [l for i, l in enumerate(layers.split(enc_out, enc_out.shape[1], 1)) if i > j], axis=1)
            r_attn = layers.matmul(indices, enc_out_r)
        else:
            r_attn = word_vec
        new_words[j] = layers.concat([word_vec, l_attn, r_attn], axis=2)

    # attn_scores = layers.concat(word_scores_removej_list, axis=1)
    # max_js = layers.concat(indices_list, axis=1)
    new_enc_out = layers.concat(new_words, axis=1)
    print(new_enc_out)
    return new_enc_out, attn_scores, max_js



def classifier_weightedAdd_all_attention(enc_out,self_attn_mask):
    # n_head_self_attn_mask = fluid.layers.stack(
    #     x=[self_attn_mask] * 1, axis=1)
    # n_head_self_attn_mask.stop_gradient = True
    attn_scores = multi_head_attention(
        pre_process_layer(enc_out,'da',0.0,name='classifier_pre_att'),
        None,None,attn_bias=self_attn_mask,
        d_key=768,d_value=768,d_model=768,n_head=1,
        dropout_rate=0.0,param_initializer=None,name= 'classifier_att_score',attention_only=True)
    # a_sent_scores = layers.split(attn_scores, attn_scores.shape[1], 1)
    # a_sent = layers.split(enc_out, enc_out.shape[1], 1)
    # new_words = [layers.create_tensor(dtype='float32')] * enc_out.shape[1]
    # max_js = layers.fill_constant(shape=[1], dtype='float32', value=0.0)
    # for j, [word_scores, word_vec] in enumerate(zip(a_sent_scores, a_sent)):
    #     new_words[j] = layers.matmul(word_scores,enc_out)
    # new_enc_out = layers.concat(new_words, axis=1)

    new_enc_out = layers.matmul(attn_scores,enc_out)
    max_js = None
    return new_enc_out, attn_scores, max_js

def classifier_weightedAdd_all_attention_concat_middle(enc_out,self_attn_mask):
    # n_head_self_attn_mask = fluid.layers.stack(
    #     x=[self_attn_mask] * 1, axis=1)
    # n_head_self_attn_mask.stop_gradient = True
    attn_scores = multi_head_attention(
        pre_process_layer(enc_out,'da',0.0,name='classifier_pre_att'),
        None,None,attn_bias=self_attn_mask,
        d_key=768,d_value=768,d_model=768,n_head=1,
        dropout_rate=0.0,param_initializer=None,name= 'classifier_att_score',attention_only=True, need_combine=False)
    # a_sent_scores = layers.split(attn_scores, attn_scores.shape[1], 1)
    # a_sent = layers.split(enc_out, enc_out.shape[1], 1)
    # new_words = [layers.create_tensor(dtype='float32')] * enc_out.shape[1]
    # max_js = layers.fill_constant(shape=[1], dtype='float32', value=0.0)
    # for j, [word_scores, word_vec] in enumerate(zip(a_sent_scores, a_sent)):
    #     new_words[j] = layers.concat([word_vec,layers.matmul(word_scores,enc_out)], axis=2)
    # new_enc_out = layers.concat(new_words, axis=1)
    to_concat = layers.matmul(attn_scores,enc_out)
    new_enc_out = layers.concat([enc_out, to_concat], 2)
    max_js = None
    print(new_enc_out)
    return new_enc_out, attn_scores, max_js