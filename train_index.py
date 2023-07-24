import argparse
import os
import pickle

from log import logger

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, default="dataset/44k", help="path to root dir"
    )
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                    help='JSON file for configuration')
    parser.add_argument(
        "--output_dir", type=str, default="logs/44k", help="path to output dir"
    )

    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    spk_dic = hps.spk
    result = {}
    
    for k,v in spk_dic.items():
        # Q: 解释一下这个 logger 到底输出了个啥
        # A: 我不知道
        logger.info("index {} feature...",k)
        index = utils.train_index(k,args.root_dir)
        result[v] = index

    with open(os.path.join(args.output_dir,"feature_and_index.pkl"),"wb") as f:
        pickle.dump(result,f)