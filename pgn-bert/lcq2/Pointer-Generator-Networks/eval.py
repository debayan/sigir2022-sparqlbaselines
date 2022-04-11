from logging import getLogger
from config import Config
from model import Model
from trainer import Trainer
from utils import init_seed
from logger import init_logger
from utils import data_preparation
import sys
from glob import glob


def train(bertpath, config, modelpath):
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    test_data, train_data, valid_data = data_preparation(config)
    model = Model(config).to(config['device'])
    trainer = Trainer(config, model, bertpath)
    test_result = trainer.evaluate(test_data, model_file=modelpath)
    #best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    #logger.info('best valid loss: {}, best valid ppl: {}'.format(best_valid_score, best_valid_result))
    #test_result = trainer.evaluate(test_data)
    #print(test_result)
    #logger.info('test result: {}'.format(test_result))


if __name__ == '__main__':
    modelpath = sys.argv[1]
    bertpath = sys.argv[2]
    config = Config(config_dict={'test_only': True})
    train(bertpath, config, modelpath)
