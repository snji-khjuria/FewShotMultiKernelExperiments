
# coding: utf-8

# In[1]:

#do the essential imports
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
# In[2]:

#hyperparameters
max_sequence_length = 30
batch_size          = 128
n_way               = 5
k_shot              = 5
lr                  = 0.001
epochs              = 200
training_location   = "./data/train_string_05_shot"
testing_location    = "./data/test_string_05_shot"


# In[3]:

#read and load the glove vectors
glove_location = "./embeddings.txt"
embedding_dim  = 50
#accessing the word-IDs from the sentence
UNK = "<unk>"
ZERO = "<zero>"
def get_glove_vectors(glove_vectors_file):
    glove_wordmap = {}
    with open(glove_vectors_file, "r") as glove:
        for line in tqdm(glove):
            name, vector = tuple(line.split(" ", 1))
            glove_wordmap[name] = np.fromstring(vector, sep=" ")
    if UNK not in glove_wordmap:
        wvecs = []
        for item in glove_wordmap.items():
            wvecs.append(item[1])
        s = np.vstack(wvecs)
        # Gather the distribution hyperparameters
        v = np.var(s,0) 
        m = np.mean(s,0) 
        RS = np.random.RandomState()
        glove_wordmap[UNK] = RS.multivariate_normal(m, np.diag(v))
    glove_wordmap[ZERO] = np.zeros_like(glove_wordmap["the"])
    vocab = glove_wordmap.keys()
    word2index = {k:i for i, k in enumerate(vocab)}
    index2word = {i:k for i, k in enumerate(vocab)}
    return glove_wordmap, word2index, index2word

print("Reading the glove vectors...")
glove_vectors, word2index, index2word = get_glove_vectors(glove_location)
print("Glove vectors read")



# In[25]:

#do the shuffling of data
def shuffle_data(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)
    a, b = zip(*combined)
    return a, b

#getting the word map from vocabulary
def sentence2sequence(sentence, word2index, max_sequence_length):
    tokens = sentence.lower().split(" ")
    rows   = []
    for token in tokens:
        i = len(token)
        while len(token)>0:
            word = token[:i]
            if word in word2index:
                rows.append(word2index[word])
                token = token[i:]
                i = len(token)
                continue
            else:
                i-=1
            if i==0:
                rows.append(word2index[UNK])
                break
    while len(rows)<max_sequence_length:
        rows.append(word2index[ZERO])
    rows = rows[:max_sequence_length]
    return np.asarray(rows)

def get_one_hot(c, total):
    vector = np.zeros(total)
    vector[c] = 1.0
    return vector


import pyximport; pyximport.install()
from string_kernel import ssk, string_kernel
#parse the complete support set

def parse_support_set(lines, word2index, max_sequence_length):
    lbda = .6
    lines      = lines[1:-1]
    test_point = lines[-1]
    #get rid of lines till -----------
    lines      = lines[:-2]
    x_hat, y_hat = test_point.strip().split("\t")
    x = []
    y = []
    for line in lines:
        x_i, y_i = line.strip().split("\t")
        x.append(x_i)
        y.append(y_i)
    str_sim = [ssk(x_i, x_hat, 4, lbda, accum=True) for x_i in x]    
    classes_in_set    = set(y+[y_hat])
    class2index       = {k:i for i, k in enumerate(classes_in_set)}
    str_sim_per_class = [0.0]*len(classes_in_set)
    for i, y_i in enumerate(y):
        class_i = class2index[y_i]
        str_sim_per_class[class_i]+=str_sim[i]
    str_sim = str_sim_per_class
    y_hat             = get_one_hot(class2index[y_hat], len(classes_in_set))
    y                 = [get_one_hot(class2index[y_i], len(classes_in_set)) for y_i in y]
    x                 = [sentence2sequence(x_i, word2index, max_sequence_length) for x_i in x]
    x_hat             = sentence2sequence(x_hat, word2index, max_sequence_length)
    x, y              = shuffle_data(x, y)
    return np.asarray(x).astype(np.int32), np.asarray(y).astype(np.int32), np.asarray(x_hat).astype(np.int32), np.asarray(y_hat).astype(np.int32), np.asarray(str_sim).astype(np.float32)
    
    
def generate_support_set_batch(file_location, batch_size, word2index, max_sequence_length):
    with open(file_location, 'r') as f:
        lines = f.readlines()
        x_support_set_batch       = []
        y_support_set_batch       = []
        str_sim_support_set_batch = []
        xhat_batch                = []
        yhat_batch                = []
        support_set = []
        for line in lines:
            line = line.strip()
            support_set.append(line)
            if line=="]":
                x_ss, y_ss, x_hat, y_hat, strsim_ss = parse_support_set(support_set, word2index, max_sequence_length)
                support_set = []
                x_support_set_batch.append(x_ss)
                y_support_set_batch.append(y_ss)
                str_sim_support_set_batch.append(strsim_ss)
                xhat_batch.append(x_hat)
                yhat_batch.append(y_hat)
                if len(xhat_batch)==batch_size:
                    yield np.asarray(x_support_set_batch).astype(np.int32), np.asarray(y_support_set_batch).astype(np.int32), np.asarray(xhat_batch).astype(np.int32), np.asarray(yhat_batch).astype(np.int32), np.asarray(str_sim_support_set_batch).astype(np.float32)
                    x_support_set_batch       = []
                    y_support_set_batch       = []
                    xhat_batch                = []
                    yhat_batch                = []
                    str_sim_support_set_batch = []


# In[27]:

#building the network model
#tf.reset_default_graph()

#finding the length of sequence
def length(sequence):
    used   = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

#do the contextual embeddings for 
def fce_g(encoded_x_i):
    fw_cell = tf.contrib.rnn.BasicLSTMCell(64) #25 is the size of embedding/2
    bw_cell = tf.contrib.rnn.BasicLSTMCell(64) #half of the embeddding size
    outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, encoded_x_i, dtype=tf.float32)
    return tf.add(tf.stack(encoded_x_i), tf.stack(outputs))

#find the contextual embeddings for xhat
def fce_f(batch_size, encoded_x, g_embedding):
    K          = 20
    cell       = tf.contrib.rnn.BasicLSTMCell(128) #50 is the embedding dimension
    prev_state = cell.zero_state(batch_size, tf.float32) # state[0] is c, state[1] is h
    for step in range(K):
        output, state = cell(encoded_x, prev_state) # output: (batch_size, 64)
        h_k = tf.add(output, encoded_x) # (batch_size, 64)
        #to compute the equation a(h_k-1, g(x_i))
        content_based_attention = tf.nn.softmax(tf.multiply(prev_state[1], g_embedding))    # (n * k, batch_size, 64)
        r_k = tf.reduce_sum(tf.multiply(content_based_attention, g_embedding), axis=0)      # (batch_size, 64)
        prev_state = tf.contrib.rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))
    return output


#cosine similarity for embedded support set and target
def cosine_similarity(target, support_set):
    target_normed  = tf.nn.l2_normalize(target, 1)
    sup_similarity = []
    for i in tf.unstack(support_set):
        #batch X 64
        i_normed = tf.nn.l2_normalize(i, 1)
        #(batch, )
        similarity = tf.matmul(tf.expand_dims(target_normed, 1), tf.expand_dims(i_normed, 2))
        sup_similarity.append(similarity)
    #batch, n*k
    return tf.squeeze(tf.stack(sup_similarity, axis=1))


def build_model(n_way, k_shot, batch_size, max_sequence_length, embedding, epochs, train_loc, test_loc):
    #create the placeholder section
    support_set_x_ph              = tf.placeholder(tf.int32, [None, n_way*k_shot, max_sequence_length])
    support_set_y_ph              = tf.placeholder(tf.int32, [None, n_way*k_shot, n_way])
    x_hat_ph                      = tf.placeholder(tf.int32, [None, max_sequence_length])
    y_hat_ph                      = tf.placeholder(tf.int32, [None, n_way])
    string_similarity_ph          = tf.placeholder(tf.float32, [None, n_way])
    
    
    
    #embeddings loading section
    init_e                        = np.asarray(list(embedding.values()), dtype=np.float32)
    W                             = tf.get_variable(name="W", initializer=init_e, trainable=False)
    embedding_init                = W
    support_set_x_embedded        = tf.nn.embedding_lookup(embedding_init, support_set_x_ph)
    x_hat_embedded                = tf.nn.embedding_lookup(embedding_init, x_hat_ph)
    
    sequences_support_set         = tf.reshape(support_set_x_embedded, [-1, max_sequence_length, embedding_dim])
    lstm_batch_input              = tf.concat([sequences_support_set, x_hat_embedded], axis=0)
    lstm_size = 128
    lstm_cell                     = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstm_output, lstm_state       = tf.nn.dynamic_rnn(lstm_cell, lstm_batch_input, dtype=tf.float32,
                                                sequence_length=length(lstm_batch_input))    
    lstm_batch_state              = lstm_state.h
    support_set_encoded, x_hat_encoded = tf.split(lstm_batch_state, [batch_size*n_way*k_shot, batch_size], axis=0)
    support_set_encoded                = tf.reshape(support_set_encoded, [batch_size, n_way*k_shot, -1])
    support_set_encoded                = tf.unstack(support_set_encoded, axis=1)
    
    g_embedding                        = fce_g(support_set_encoded)
    
    f_embedding                        = fce_f(batch_size, x_hat_encoded, g_embedding)
    embedding_similarity               = cosine_similarity(f_embedding, g_embedding)
    attention                 = tf.nn.softmax(embedding_similarity)
    y_hat                     = tf.matmul(tf.expand_dims(attention, 1), tf.cast(support_set_y_ph, tf.float32))
    mnet_logits               = tf.squeeze(y_hat)
    print("mnet logits shape is ")
    print(mnet_logits)
    print("string similarity ph is")
    print(string_similarity_ph)
    str_sim_with_attention    = tf.concat([tf.expand_dims(mnet_logits, 2), tf.expand_dims(string_similarity_ph, 2)], axis=2)
    str_sim_with_attention    = tf.reshape(str_sim_with_attention, shape=[-1, 2])
    weight_initer          = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    W_sim = tf.get_variable(name="Weight", dtype=tf.float32, shape=[2, 1], initializer=weight_initer)
    logits = tf.matmul(str_sim_with_attention, W_sim)
    logits = tf.reshape(logits, shape=[-1, n_way])
    pred                 = tf.argmax(logits, axis=1)
    loss_op              = tf.nn.softmax_cross_entropy_with_logits(labels=y_hat_ph, logits=logits)
    train_op             = tf.train.AdamOptimizer(lr).minimize(loss_op)
    
#     self.logits               = tf.squeeze(y_hat)
#     self.pred                 = tf.argmax(self.logits, 1)
        
    
    
    #build lstm layer on top of it
    
    #do the printing section
    print("Support set x placeholder")
    print(support_set_x_ph)
    print("Support set y placeholder")
    print(support_set_y_ph)
    print("Xhat placeholder")
    print(x_hat_ph)
    print("Yhat placeholder")
    print(y_hat_ph)
    print("String similarity placeholder")
    print(string_similarity_ph)
    print("embedding init")
    print(embedding_init)
    print("Support set embedded")
    print(support_set_x_embedded)
    print("xhat embedded")
    print(x_hat_embedded)

    print("Sequences support set ")
    print(sequences_support_set)
    print("LSTM batch size")
    print(lstm_batch_state)
    print("Suppot set encoded is ")
    print(len(support_set_encoded))
    print(support_set_encoded)
    print("x hat encoded")
    print(x_hat_encoded)
    print("f embeddings")
    print(f_embedding)
    print("g embedding")
    print(g_embedding)
    print("Similarity is ")
    print(embedding_similarity)
    print("Attention size is ")
    print(attention)
    print("Attention of string similarity")
    print(str_sim_with_attention)
    print("weights are ")
    print(W_sim)
    print("logits are ")
    print(logits)
    print("predictions shape is ")
    print(pred)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        count = 0
        for support_set in generate_support_set_batch(training_location, batch_size, word2index, max_sequence_length):
            count+=1
            if count%100==0:
                print(str(count) + "processed")
                loss_total = []
                correct_count = 0
                total_count = 0
                for support_set2 in generate_support_set_batch(testing_location, batch_size, word2index, max_sequence_length):
                    x_supp, y_supp, x_hat, y_hat, str_sim = support_set2
                    feed_dict = {support_set_x_ph:x_supp, support_set_y_ph:y_supp, x_hat_ph:x_hat, y_hat_ph:y_hat, string_similarity_ph:str_sim}
                    loss_ans, pred_ans = sess.run([loss_op, pred], feed_dict=feed_dict)
#                     print("loss ans is " + str(loss_ans))	
                    loss_total.extend(loss_ans)
                    actual     = np.asarray([list(y_hat_i).index(1) for y_hat_i in y_hat])
                    correct_count+=np.sum(pred_ans == actual)
                    total_count+=pred_ans.shape[0]
#                     print(str(correct_count) +" out of " + str(total_count))
                loss_total = np.asarray(loss_total)
                accuracy = (float(correct_count)/float(total_count))*100.0
                print("Loss is "+str(np.mean(loss_total))+" correct "+str(correct_count)+" out of "+str(total_count)+" accuaracy "+str(accuracy))
            x_supp, y_supp, x_hat, y_hat, str_sim = support_set
            feed_dict = {support_set_x_ph:x_supp, support_set_y_ph:y_supp, x_hat_ph:x_hat, y_hat_ph:y_hat, string_similarity_ph:str_sim}    
            _ = sess.run(train_op, feed_dict=feed_dict)
        
        loss_total = []
        for support_set in generate_support_set_batch(training_location, batch_size, word2index, max_sequence_length):
            x_supp, y_supp, x_hat, y_hat, str_sim = support_set
            feed_dict = {support_set_x_ph:x_supp, support_set_y_ph:y_supp, x_hat_ph:x_hat, y_hat_ph:y_hat, string_similarity_ph:str_sim}    
            loss_ans = sess.run([loss_op], feed_dict=feed_dict)
            loss_total.extend(loss_ans)
        loss_total = np.asarray(loss_total)
        print("Loss is " + str(np.mean(loss_total)))
build_model(n_way, k_shot, batch_size, max_sequence_length, glove_vectors, epochs, training_location, testing_location)


# In[ ]:



