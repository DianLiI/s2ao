import tensorflow as tf
from s2ao.preprocessing import DataLoader, BatchGenerator
from s2ao.models import s2ao
import s2ao.config as cfg
from tqdm import tqdm
import logging
import sys


def train_s2ao():
    config = cfg.get_config()
    level = config["logging"]["level"].lower()
    if level == "debug":
        l = logging.DEBUG
    else:
        l = logging.INFO
    logging.basicConfig(format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]\n %(message)s", level=l)
    logger = logging.getLogger(__name__)
    fileHandler = logging.FileHandler("{0}/{1}.log".format(config["path"]["log_dir"], "new_model"))
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
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
    saver = tf.train.Saver(max_to_keep=3)
    while True:
        for feats, actions, objects in tqdm(generator.get_epoch("train"), total=generator.get_num_batches("train")):
            model.train(feats, actions, objects)
            print(model.predict(feats))
        # n_epoch += 1
        # logger.info("epoch: {}".format(n_epoch))
        # feats, actions, objects = generator.get_all("valid")
        # cost_v, acc_v, cost_n, acc_n = model.actions(feats, verbs, nouns)
        # logger.info("valid\nverb cost: {:.5f}\nnoun cost: {:.5f}\nverb accuracy: {:.5f}\nnoun accuracy: {:.5f}")
        # logger.info("verb cost: {:.5f}".format(cost_v))
        # logger.info("noun cost: {:.5f}".format(cost_n))
        # logger.info("verb accuracy: {:.5f}".format(acc_v))
        # logger.info("noun accuracy: {:.5f}".format(acc_n))
        # saver.save(model.sess, 'model.net', global_step=n_epoch)
        # if n_epoch % 5 == 0:
        #     feats, verbs, nouns = generator.get_all("test")
        #     cost_v, acc_v, cost_n, acc_n = model.stats(feats, verbs, nouns)
        #     logger.info("test")
        #     logger.info("verb cost: {:.5f}".format(cost_v))
        #     logger.info("noun cost: {:.5f}".format(cost_n))
        #     logger.info("verb accuracy: {:.5f}".format(acc_v))
        #     logger.info("noun accuracy: {:.5f}".format(acc_n))
        continue
if __name__ == "__main__":
    train_s2ao()

