from s2ao.preprocessing import DataLoader, BatchGenerator
from s2ao.models import s2ao
from s2ao.config import config
from tqdm import tqdm
import logging
import sys


def train_s2ao():
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger(__name__)
    level = config["logging"]["level"].lower()
    if level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(config["path"]["log_dir"], "new_model"))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger online")

    loader = DataLoader(config)
    loader.load_data()
    batch_size = int(config["training"]["batch_size"])
    hidden1 = int(config["training"]["hidden_size1"])
    hidden2 = int(config["training"]["hidden_size2"])
    generator = BatchGenerator(loader, batch_size)
    model = s2ao(hidden1, hidden2, loader.longest_sequence, loader.action_size, loader.object_size)

    n_epoch = 0
    while True:
        for feats, verbs, nouns in tqdm(generator.get_epoch("train"), total=generator.get_num_batches("train")):
            model.train(feats, verbs, nouns)
        n_epoch += 1
        logger.info("epoch: {}".format(n_epoch))
        logger.info("train")
        # print("verb cost: {:.5f}".format())
        # print("noun cost: {:.5f}".format())
        logger.info("verb accuracy: {:.5f}".format(model.t_v / (model.t_v + model.f_v)))
        logger.info("noun accuracy: {:.5f}".format(model.t_n / (model.t_n + model.f_n)))
        model.t_v, model.f_v, model.t_n, model.f_n = 0, 0, 0, 0
        feats, verbs, nouns = generator.get_all("valid")
        cost_v, acc_v, cost_n, acc_n = model.stats(feats, verbs, nouns)
        logger.info("valid")
        logger.info("verb cost: {:.5f}".format(cost_v))
        logger.info("noun cost: {:.5f}".format(cost_n))
        logger.info("verb accuracy: {:.5f}".format(acc_v))
        logger.info("noun accuracy: {:.5f}".format(acc_n))
        if n_epoch % 10 == 0:
            feats, verbs, nouns = generator.get_all("test")
            cost_v, acc_v, cost_n, acc_n = model.stats(feats, verbs, nouns)
            logger.info("test")
            logger.info("verb cost: {:.5f}".format(cost_v))
            logger.info("noun cost: {:.5f}".format(cost_n))
            logger.info("verb accuracy: {:.5f}".format(acc_v))
            logger.info("noun accuracy: {:.5f}".format(acc_n))
        continue

