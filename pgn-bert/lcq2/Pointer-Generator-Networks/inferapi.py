from logging import getLogger
from config import Config
from model import Model
from trainer import Trainer
from utils import init_seed
from logger import init_logger
from utils import data_preparation
import json
import sys
import copy
from flask import request, Response
from flask import Flask
from gevent.pywsgi import WSGIServer


def init():
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    model = Model(config).to(config['device'])
    return model

def infer(model,config, d):
    test_data, _, _ = data_preparation(config, d['linkedentrelstring'],d['linkedentrelvecs'])
    trainer = Trainer(config, model)
    test_result = trainer.evaluate_single(test_data, model_file=config['load_experiment'])
    return test_result

#config = Config(config_dict={'test_single': True,'load_experiment': 'saved_ppspq1/Fire-At-Dec-14-2021_16-51-40.pth'})
#config = Config(config_dict={'test_single': True,'load_experiment': 'saved_ppspq4/Fire-At-Dec-18-2021_23-34-52.pth'})
config = Config(config_dict={'test_single': True,'load_experiment': 'saved_ppspq6/Fire-At-Dec-29-2021_21-29-13.pth'})
app = Flask(__name__)
print("listening ...")

@app.route('/generatequery', methods=['POST'])
def generatequery():
    d = request.get_json(silent=True)
    citem = copy.deepcopy(d)
    model = init()
    result = infer(model,config,d)
    del model
    #print("result:",result)
    citem['predicted_query'] = result
    del citem['linkedentrelvecs']
    return json.dumps(citem, indent=4)

if __name__ == '__main__':
    
    http_server = WSGIServer(('', 2224), app)
    http_server.serve_forever()
