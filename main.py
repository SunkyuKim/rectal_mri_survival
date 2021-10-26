import torch
import nltk
import argparse
import utils
import models
import numpy as np
import pandas as pd
import datetime
import time
import json
import itertools
import collections
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lifelines.utils import concordance_index
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import setproctitle

from tensorboardX  import SummaryWriter


def main(config):
    config.use_bert = True if "bert" in str(config.model).lower() else False

    config.result_dir = "{}/{}_{}/".format(config.result_dir, config.model, config.split_seed)
    os.makedirs(config.result_dir, exist_ok=True)

    start_time = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    try:
        logger = utils.BKLogger.logger(config.result_dir, start_time)
    except:
        logger = utils.BKLogger.logger()


    emr = utils.EMRParser("./sample_data.json",
                            max_length=512,
                            use_cache=True,
                            cv_split=5,
                            use_bert=config.use_bert,
                            bert_pretrained=config.pretrained,
                            result_dir=config.result_dir,
                            split_seed=config.split_seed)

    emr.dim_demo_in = len(emr.traindf['demo'].iloc[0][0])
    logger.info("Converting Data to Tensor..")

    traindata = utils.SurvivalVisitDataset(emr.traindf)
    testdata = utils.SurvivalVisitDataset(emr.testdf)

    traindl = DataLoader(dataset=traindata, batch_size=config.batch_size, shuffle=True)
    traindl_noshuffle = DataLoader(dataset=traindata, batch_size=config.batch_size, shuffle=False)
    testdl = DataLoader(dataset=testdata, batch_size=config.batch_size, shuffle=False)

    #setup model
    if config.model== "VisitBertCox":
        model = models.VisitBertCox.from_pretrained(config.pretrained,
                                                from_tf=config.from_tf,
                                                dim_demo_in=emr.dim_demo_in,
                                                only_first_report=config.only_first_report)
    elif config.model== "AttnLSTMCox":
        model = models.AttnLSTMCox(vocab_size=vocab_size)
    else:
        raise Exception

    config.device = torch.device("cuda:"+str(config.device)) if torch.cuda.is_available() else torch.device("cpu")
    model.to(config.device)

    loss_fn = models.PartialNLL()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2reg)

    performance_fw = open(config.result_dir + "/performance.txt", "w")
    performance_fw.write("\t".join(["Ep", "train", "test"]) + "\n")
    performance_fw.flush()

    train_maxCindex = 0
    test_maxCindex = 0
    for epoch in range(1, config.epochs+1):

        logger.info("====\tEpoch: {}".format(epoch))
        train(config, traindl, model, loss_fn, optimizer)

        train_logdict = evaluate(config, traindl_noshuffle, model)
        test_logdict = evaluate(config, testdl, model)
        logger.info("Train\tC-index : {:.3f}".format(train_logdict["Cindex"]))
        logger.info("Test\tC-index : {:.3f}".format(test_logdict["Cindex"]))

        performance_fw.write("\t".join([str(epoch), str(train_logdict["Cindex"]), str(test_logdict["Cindex"])])+"\n")
        performance_fw.flush()

        if train_logdict["Cindex"] > train_maxCindex:
            train_maxCindex = train_logdict["Cindex"]
            with open("{}/visualdict_trainmax.pkl".format(config.result_dir, epoch), "wb") as fw:
                writedict = {   "train":train_logdict,
                                "test":test_logdict,
                                "epoch":epoch}
                pickle.dump(writedict, fw)

        if test_logdict["Cindex"] > test_maxCindex:
            test_maxCindex = test_logdict["Cindex"]
            with open("{}/visualdict_testmax.pkl".format(config.result_dir, epoch), "wb") as fw:
                writedict = {   "train":train_logdict,
                                "test":test_logdict,
                                "epoch":epoch}
                pickle.dump(writedict, fw)

        # save best train / best test / last
        #torch.save(model.state_dict(), "{}/ep{}.ckpt".format(config.result_dir, epoch))

    performance_fw.close()

    return


def train(config, dl, model, loss_fn, optimizer):
    return _run(config, dl, model, loss_fn, optimizer)


def evaluate(config, dl, model):
    return _run(config, dl, model)


def _run(config, dl, model, loss_fn=None, optimizer=None):
    if (loss_fn is not None) and (optimizer is not None):
        model.train()
    else:
        model.eval()

    logdict = collections.defaultdict(list)

    for it, batch_data in enumerate(tqdm(dl, desc="Iteration")):
        for i,data in enumerate(batch_data):
            if isinstance(data, torch.Tensor):
                batch_data[i] = data.to(config.device)
        batch_x, batch_x_demo, batch_y, batch_censored, batch_idx, batch_visit_len = batch_data
        batch_idx = batch_idx.cpu().numpy()
        batch_R = torch.tensor(dl.dataset.R[batch_idx,:][:,batch_idx]).to(config.device)
        mask_x = (batch_x > 0).byte()

        if model.training:
            theta, att, patientemb = model(batch_x, demo=batch_x_demo, attention_mask=mask_x)
            optimizer.zero_grad()
            loss = loss_fn(theta[-1], batch_R, batch_censored).cpu()
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                theta, att, patientemb = model(batch_x, demo=batch_x_demo, attention_mask=mask_x)

        logdict["y"] += batch_y.tolist()
        logdict["c"] += batch_censored.tolist()
        logdict["theta"] += (-torch.stack(theta, 1)).tolist()
        logdict["final_theta"] += (-theta[-1]).tolist()
        #logdict["attlist"] += att.tolist()
        logdict["patientemblist"] += patientemb.tolist()

    event_observed = np.array([1 if v==0 else 0 for v in logdict["c"]])
    cindex = concordance_index(logdict["y"], logdict["final_theta"], event_observed)
    logdict["Cindex"] = cindex

    return logdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="VisitBertCox")
    parser.add_argument("--pretrained", type=str, default="vocab/kexin_clinicalbert/pretraining")
    #parser.add_argument("--pretrained", type=str, default="vocab/biobert_v1.1_pubmed")
    #parser.add_argument("--pretrained", type=str, default="vocab/cased_L-12_H-768_A-12")
    parser.add_argument("--from_tf", action='store_true')

    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=int)
    parser.add_argument('--l2reg', default=3e-7, type=int)
    parser.add_argument("--processname", type=str, default="EMRsurvival")

    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--torch_seed", type=int, default=0)

    parser.add_argument("--only_first_report", action='store_true')
    parser.add_argument("--result_dir", type=str, default="./results/")
    config = parser.parse_args()

    setproctitle.setproctitle(config.processname)

    import copy
    for i in range(1):
        tmpconfig = copy.deepcopy(config)
        tmpconfig.split_seed = i
        main(tmpconfig)
