import numpy as np
import json
import logging
import os
import spacy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pickle
import tqdm
import re
import collections

# Multiprocessing
# import parmap
import multiprocessing
from itertools import repeat
#logger = None

import transformers

class EMRParser():
    """
    parse the emr dataset already parsed by yonghwa
    """
    def __init__(self, datapath, max_length=None, use_cache=False, cv_split=5, use_bert=False, bert_pretrained="bert-base-uncased", result_dir="", split_seed=0, maxlen=4):
        self.logger = BKLogger.logger()
        self.result_dir = result_dir
        self.split_seed = split_seed
        self.maxlen = maxlen

        cache_path = "{}/cache.pickle".format(self.result_dir)

        if use_cache:
            self.logger.debug("loading cached data..")
            try:
                with open(cache_path, "rb") as fr:
                    # self.traindf, self.testdf, self.vocab_size = pickle.load(fr)
                    self.traindf, self.testdf = pickle.load(fr)
                    self.logger.debug("#train:{}\tcensored:{}".format(self.traindf.shape[0], self.traindf.censored.sum()))
                    self.logger.debug("#test:{}\tcensored:{}".format(self.testdf.shape[0], self.testdf.censored.sum()))
                return
            except FileNotFoundError:
                print("Not found cache")

        self.logger.debug("loading raw data..")
        with open(datapath) as fr:
            self.rawdata = json.load(fr)

        self.num_cores = int(multiprocessing.cpu_count()*0.5)

        if use_bert:
            self.bert_vocab_name = bert_pretrained
            self.bert_vocab_path = self.bert_vocab_name
            #os.makedirs(self.bert_vocab_path, exist_ok=True)
            self.logger.debug("constructing vocab..")
            self.build_bert_vocab()
            self.max_length_threshold = max_length
            self.bert_text2idx()
        else:
            self.logger.debug("start data parsing")
            self.logger.debug("loading spacy..")
            self.spacy_nlp = spacy.load('en_core_web_sm')
            self.max_length_threshold = max_length
            self.build_vocab()
            self.logger.debug("vocab:{}".format(self.vocab.idx))
            self.logger.debug("constructing text2idx..")
            self.text2idx()
            self.vocab_size = len(self.vocab)

        self.logger.debug("making dataframe.. ")
        self.datadf = pd.DataFrame(self.data)
        self.datadf['survival_days'] = self.datadf.survival_days.astype('int64')
        self.datadf['censored'] = [1 if v=="True" else 0 for v in self.datadf.censored]

        ### demographics ###
        demo_list = list()
        #demo_features = ['age', 'gender', 'CEA']
        demo_features = ['age', 'gender', 'CEA', 'OP', 'RT']

        for i in range(self.datadf.shape[0]):
            row = self.datadf[demo_features].iloc[i]
            tmp_demo = list()
            for feat in demo_features:
                if isinstance(row[feat], list):
                    tmp_demo.extend(row[feat])
                else:
                    tmp_demo.append(row[feat])
            demo_list.append([float(re.sub("[^0-9\.]", "", v)) for v in tmp_demo])

        self.datadf['demo'] = demo_list
        self.logger.debug("#samples:{}, censored:{}".format(self.datadf.shape[0], self.datadf['censored'].sum()))


        ### transform datadf to visit_datadf ###
        datadf_dict = collections.defaultdict(list)
        groups = self.datadf.groupby('research_no')
        for i,groupdf in groups:
            #datadf_dict['research_no'].append(i)
            groupdf = groupdf.sort_values('survival_days', ascending=False)
            for col in groupdf.columns:
                if col == "survival_days":
                    datadf_dict[col].append(groupdf[col].tolist()[0])
                elif col == "censored":
                    datadf_dict[col].append(groupdf[col].tolist()[-1])
                else:
                    datadf_dict[col].append(groupdf[col].tolist()[:self.maxlen])
        self.datadf = pd.DataFrame(datadf_dict)

        print(self.datadf.head())

        self.datadf['multiple_visit'] = self.datadf['reading_text'].apply(lambda x : 1 if len(x)>1 else 0)

        # No stratified about censored
        """
        train_idx, test_idx = train_test_split(list(range(self.datadf.shape[0])),
                                                test_size=0.2,
                                                random_state=0)
        """
        # Stratified about censored
        """
        1. multiple_visit
        2. censored
        """
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.split_seed)
        for train_idx, test_idx in sss.split(range(self.datadf.shape[0]), self.datadf['multiple_visit']):
            self.traindf = self.datadf.iloc[train_idx]
            self.logger.debug("#train:{}\tcensored:{}".format(self.traindf.shape[0], self.traindf.censored.sum()))

            self.testdf = self.datadf.iloc[test_idx]
            self.logger.debug("#test:{}\tcensored:{}".format(self.testdf.shape[0], self.testdf.censored.sum()))


        cv_sss = StratifiedShuffleSplit(n_splits=cv_split, test_size=0.2, random_state=self.split_seed)
        self.cv_dfs = []

        for cv_train_idx, cv_valid_idx in cv_sss.split(range(self.traindf.shape[0]), self.traindf['multiple_visit']):
            self.cv_dfs.append({
                'cv_traindf': self.traindf.iloc[cv_train_idx],
                'cv_validdf': self.traindf.iloc[cv_valid_idx]
            })

        for cv_idx in range(len(self.cv_dfs)):
            self.logger.debug("\t{}th_#cv_train:{}\tcensored:{}".format(cv_idx, self.cv_dfs[cv_idx]['cv_traindf'].shape[0], self.cv_dfs[cv_idx]['cv_traindf'].censored.sum()))
            self.logger.debug("\t{}th_#cv_valid:{}\tcensored:{}".format(cv_idx, self.cv_dfs[cv_idx]['cv_validdf'].shape[0], self.cv_dfs[cv_idx]['cv_validdf'].censored.sum()))

        # self.vocab_size = self.vocab.idx

        with open("{}/cache.pickle".format(self.result_dir), "wb") as fw:
            # pickle.dump((self.traindf, self.testdf, self.vocab_size), fw)
            pickle.dump((self.traindf, self.testdf), fw)

        self.logger.debug("data parsing and caching Done.")
        self.logger.debug("training data : {} , test data : {}".format(self.traindf.shape, self.testdf.shape))


    def nlp(self, text):
        text = re.sub("[\(\-\+\)]", lambda x : " {} ".format(x.group()), text)
        text = re.sub(" +", " ", text)

        return self.spacy_nlp(text)

    ##def parse_single_data(self, d, words, lengths):
    def parse_single_data(self, d):
        text = d['reading_text']
        tokens = [str(v) for v in self.nlp(text)]
        if len(tokens) == 1:
            self.logger.debug("SINGLE TEXT :\n" + text)

        ##lengths.append(len(tokens))

        # if len(tokens) > self.max_length:
        # if len(tokens) > res['max_length']:
            # self.max_length = len(tokens)
            # res['max_length'] = len(tokens)

        words = list()
        for t in tokens:
            # res['vocab'].add_word(t)
            if t not in words:
                words.append(t)
            # self.vocab.add_word(t)
        return len(tokens), words

    def build_bert_vocab(self):
        try:
            self.logger.debug("Loading bert tokenizer in " + self.bert_vocab_path)
            self.tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_vocab_path)
        except:
            self.logger.debug("Not found tokenizer - try download " + self.bert_vocab_name)
            try:
                self.tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_vocab_name)
                self.tokenizer.save_pretrained(self.bert_vocab_path)
            except:
                self.logger.debug("Cannot download tokenizer model")
                raise


        self.max_length = 0

        lengths = list()

        for d_i in tqdm.trange(len(self.rawdata)):
            d = self.rawdata[d_i]
            text = d['reading_text']
            string = "".join([" "+str(v) if v>127 else chr(v) for v in text.encode("utf8")])
            encoded = self.tokenizer.encode(string)
            #print(encoded)
            lengths.append(len(encoded))

        self.max_length = max(lengths)


    def build_vocab(self, use_cache=False, se_visit=False):
        if use_cache:
            with open("{}/cache_vocab.pickle".format(self.result_dir), "rb") as fr:
                cache = pickle.load(fr)
                self.vocab = cache['vocab']
                self.max_length = cache['max_length']
                self.logger.debug("cache_vocab loaded")
                return

        self.max_length = 0

        lengths = list()
        words = list()

        #for map_result in map_results:
        for d_i in tqdm.trange(len(self.rawdata)):
            d = self.rawdata[d_i]
            map_result = self.parse_single_data(d)
            lengths.append(map_result[0])
            words.extend(map_result[1])


        self.vocab = Vocabulary(words)
        self.max_length = max(lengths)

        cache = {'vocab': self.vocab,
                'max_length': self.max_length}
        with open("{}/cache_vocab.pickle".format(self.result_dir), "wb") as fw:
            pickle.dump(cache, fw)

    #def single_text2idx(self, d, res):
    def single_text2idx(self, d):
        text = d['reading_text'].strip()
        tokens = [str(v) for v in self.nlp(text)]
        if self.max_length_threshold is not None:
            if len(tokens) > self.max_length_threshold:
                tokens = tokens[:self.max_length_threshold]
            else:
                num_pad = self.max_length_threshold-len(tokens)
                tokens = tokens + ['<pad>']*num_pad
        idx_list = [self.vocab(str(t)) for t in tokens]
        d['text2idx'] = idx_list
        # self.data.append(d)
        #res.append(d)
        return d


    def bert_text2idx(self):
        try:
            with open("{}/cache_bert_text2idx.pickle".format(self.result_dir), "rb") as fr:
                self.data = pickle.load(fr)
                self.logger.debug("cache_text2idx loaded")
                return
        except:
            pass

        res = list()
        if self.max_length_threshold is not None:
            max_v = self.max_length_threshold
        else:
            max_v = self.max_length
        for d_i in tqdm.trange(len(self.rawdata)):
            d = self.rawdata[d_i]
            idx_list = self.tokenizer.encode_plus(d['reading_text'], max_length=max_v, pad_to_max_length=True)
            idx_list = idx_list['input_ids']
            d['text2idx'] = idx_list
            res.append(d)

        self.data = list(res)


        with open("{}/cache_text2idx.pickle".format(self.result_dir), "wb") as fw:
            pickle.dump(self.data, fw)



    # def text2idx(self, max_length=None, use_visit=False):
    def text2idx(self):
        try:
            with open("{}/cache_text2idx.pickle".format(self.result_dir), "rb") as fr:
                self.data = pickle.load(fr)
                self.logger.debug("cache_text2idx loaded")
                return
        except:
            pass

        #self.data = list()
        res = list()
        for d_i in tqdm.trange(len(self.rawdata)):
            d = self.rawdata[d_i]
            res.append(self.single_text2idx(d))

        self.data = list(res)

        with open("{}/cache_text2idx.pickle".format(self.result_dir), "wb") as fw:
            pickle.dump(self.data, fw)




class SurvivalDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data.text2idx.tolist())
        self.x_demo = torch.tensor(data.demo.tolist())
        self.y = torch.tensor(data.survival_days.tolist()).type(torch.DoubleTensor)
        self.c = torch.tensor(data.censored.tolist()).type(torch.DoubleTensor)
        self.rawidx = torch.tensor(data.index.tolist())

        #self.x, self.x_demo, self.y, self.c = data
        self.R = self._make_R(self.y)
        self.R = torch.DoubleTensor(self.R)

        self.indices = torch.ByteTensor(list(range(self.x.shape[0])))

        #print(self.x.shape, self.y.shape, self.R.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.x_demo[idx], self.y[idx], self.c[idx], self.indices[idx], self.rawidx[idx] # indices -> for get R matrix

    def _make_R(self, y):
        idx = np.argsort(y)
        tmpR = list()
        tmpZ = [0]*y.shape[0]
        for i in idx:
            tmpZ[i] += 1.0
            tmpR.append(np.array(tmpZ))
        tmpR = np.array(tmpR)
        return tmpR[np.argsort(idx)]


class SurvivalVisitDataset(Dataset):
    def __init__(self, data):
        """
        Aggregate same research_no samples
        """
        self.x = list()
        self.x_demo = list()
        self.y = list()
        self.c = list()

        self.x = pad_sequence(data['text2idx'].apply(torch.tensor).tolist(), batch_first=True)
        self.x_demo = pad_sequence(data['demo'].apply(torch.tensor).tolist(), batch_first=True)
        self.y = torch.DoubleTensor(data['survival_days'].tolist())
        self.c = torch.DoubleTensor(data['censored'].tolist())
        self.visit_len = torch.tensor(data['text2idx'].apply(len).tolist())

#        groups = data.groupby('research_no')
#        for i,groupdf in groups:
#            groupdf = groupdf.sort_values('survival_days', ascending=False)
#            self.x.append(torch.tensor(groupdf['text2idx'].tolist()))
#            self.x_demo.append(torch.tensor(groupdf['demo'].tolist()))
#            self.y.append(groupdf['survival_days'].tolist()[0])
#            self.c.append(groupdf['censored'].tolist()[-1])
#
#        self.visit_len = torch.tensor([len(v) for v in self.x])
#        self.x = pad_sequence(self.x, batch_first=True)
#        self.x_demo = pad_sequence(self.x_demo, batch_first=True)
#        self.y = torch.DoubleTensor(self.y)
#        self.c = torch.DoubleTensor(self.c)

        #print(self.x)
        self.R = self._make_R(self.y)

        self.indices = list(range(self.x.shape[0]))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.x_demo[idx], self.y[idx], self.c[idx], self.indices[idx], self.visit_len[idx] # indices -> for get R matrix


    def _make_R(self, y):
        idx = np.argsort(y)
        tmpR = list()
        tmpZ = [0]*y.shape[0]
        for i in idx:
            tmpZ[i] += 1.0
            tmpR.append(np.array(tmpZ))
        tmpR = np.array(tmpR)
        return tmpR[np.argsort(idx)]


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    # def __init__(self):
    #     self.word2idx = {}
    #     self.idx2word = {}
    #     self.idx = 0
    #
    #     self.add_word('<pad>')
    #     self.add_word('<unk>')

    """ Initialize with words - for multiprocessing """
    def __init__(self, words):
        words = ['<pad>', '<unk>'] + words
        self.word2idx = {words[idx]: idx for idx in range(len(words))}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = max(self.word2idx.values())+1

        self.add_word('<pad>')
        self.add_word('<unk>')

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class BKLogger(object):
    __logger = None

    @classmethod
    def __getLogger(cls):
        return cls.__logger

    @classmethod
    def logger(cls, logdir, logname):
        cls.__logger = cls.__setlogger(logdir, logname)
        cls.logger = cls.__getLogger
        return cls.__logger

    @classmethod
    def __setlogger(cls, logdir, logname):
        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler = logging.FileHandler("{}/{}.log".format(logdir, logname))
        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)

        return logger


if __name__ == "__main__":
    logger = BKLogger.logger("logs", "dataparsing")
    EMRParser("data/overall_survival_data.json")
