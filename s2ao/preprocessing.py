import logging
import os
import pickle as pkl
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import s2ao.config as cfg
import s2ao.utils as utils


class DataLoader(object):
    def __init__(self, config):

        data_config = config["data"]
        self.data_root = config["path"]["data_root"]
        # TODO add support to read split from file
        if data_config["phase"] == "train":
            self.load_train, self.load_valid, self.load_test = True, True, True
        else:
            self.load_train, self.load_valid, self.load_test = True, True, True

        if data_config["feature_type"] == "both":
            self.feature_interval = (0, 4096 * 2)
        elif data_config["feature_type"] == "rgb":
            self.feature_interval = (0, 4096)
        else:
            self.feature_interval = (4096, 4096 * 2)
        self.consider_others = data_config["consider_others"] == "True"
        self.each_action = int(data_config["each_action"])
        self.action_set = data_config["include_actions"].split(", ")
        self.avoid_action_set = data_config["avoid_actions"].split(", ")
        self.simple_test = data_config["simple_test"] == "True"
        self.total_size = int(data_config["total_size"])
        self.train_weight = float(data_config["train_weight"])
        self.test_weight = float(data_config["test_weight"])
        self.valid_weight = float(data_config["valid_weight"])
        self.max_video_length = int(data_config["max_video_length"])
        self.hdf5_file = config["path"]["hdf5_file"]
        self.pkl_file = config["path"]["pkl_file"]

        # logging
        cfg = '\n'.join("%s: %s" % item for item in vars(self).items())
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loader config: {}\n".format(cfg))

        self.feats = {}
        self.labels = {}
        self.action_index = None
        self.object_index = None
        self.longest_sequence = None
        self.action_size = None
        self.object_size = None
        self.config = config
        self.is_loaded = False

    def _padding_feat(self, list_of_vect):
        a = np.zeros((self.max_video_length, self.feature_interval[1] - self.feature_interval[0])).astype(
            np.float32)
        if len(list_of_vect) < self.max_video_length:
            a[:len(list_of_vect), :] = np.array(list_of_vect, dtype=np.float32)
        else:
            a = np.array(list_of_vect[:self.max_video_length], dtype=np.float32)
        return a

    def get_data_size(self, subset):
        if subset is None:
            return self.get_data_size("train") + self.get_data_size("test") + self.get_data_size("valid")
        return len(self.labels[subset])

    def load_data(self, capoffset=0):

        self.is_loaded = True

        if os.path.exists(self.hdf5_file):
            with h5py.File(self.hdf5_file, "r") as f:
                self.feats["train"] = np.array(f.get("train/feature"))
                self.labels["train"] = np.array(f.get("train/label"), dtype=np.int32)
                self.feats["test"] = np.array(f.get("test/feature"))
                self.labels["test"] = np.array(f.get("test/label"), dtype=np.int32)
                self.feats["valid"] = np.array(f.get("valid/feature"))
                self.labels["valid"] = np.array(f.get("valid/label"), dtype=np.int32)
            with open(self.pkl_file, "rb") as f:
                self.action_index, self.action_size, self.object_index, self.object_size, self.longest_sequence = \
                    pkl.load(f)
            return

        feature_path = os.path.join(self.data_root, "paravideoclips/allfeats/")
        captionfile = open(os.path.join(self.data_root, "exportable_action_objects_youtube.csv"))
        _captionsoverall = []
        self.logger.info("Loading csv file...")
        for s in captionfile:
            _captionsoverall.append(['_'.join(s.split(',')[:3]), s.replace(',\n', '').replace('\n', '').split(',')[3:]])
        captionsoverall = [[sig, cap] if len(cap) > 1 else [sig, [cap[0], '']] for sig, cap in _captionsoverall]
        self.logger.debug("First 10 lines: {}".format(captionsoverall[:10]))
        captionsoverall = dict(captionsoverall)
        self.logger.info("Done loading csv")

        # make sure each class has almost same amount of data

        if self.simple_test:
            self.logger.info("Using 10% of the data")
            self.total_size = int(self.total_size * 0.1)
            self.each_action = int(self.each_action * 0.1)

        # cut, stir is fine, brush is left/right action; put set == push forward ? remove = pulling back?
        # add is hard to define,
        # actionset = 'put, add, cut, set, spread, remove, stir, brush'.replace(' ','').split(',')
        # actionset = 'remove, put, cut, brush, stir'.replace(' ','').split(',') # suppose to be take ??#
        # place == put == add == set?? slice == cut? chop can be down forward then down
        # chop, place, cover, combine, squeeze, mix, pour slice, fold, dip, discard, wipe']
        # old :
        # actionset = ['put', 'mix', 'cut', 'set', 'cover', 'chop', 'remove', 'place', 'stir', 'combine']

        _bound_by_each_class = (len(self.action_set) + int(self.consider_others)) * self.each_action
        _bound_by_total = self.total_size
        if _bound_by_total < 0:
            data_bound = _bound_by_each_class
        elif _bound_by_each_class < 0:
            data_bound = _bound_by_total
        else:
            data_bound = min(_bound_by_total, _bound_by_each_class)

        self.logger.info("Loading features...")
        cap_feat = []
        cap_label = []
        object_vocab = set()
        action_vocab = set()
        object_count_total = {}
        object_count_used = {}
        cap_count_total = {}
        cap_count_used = {}
        data_count = 0
        longest_sequence = 0
        file_list = os.listdir(feature_path)
        # progress bar
        for file in tqdm(file_list):
            if 0 <= data_bound <= data_count:
                break
            if file.endswith('mp4.txt') or file.endswith('webm.txt'):  # or files.endswith('avi.txt'):
                file_signature = file[file.index('_') + 1:].replace('.mp4.txt', '').replace('.webm.txt', '')
                if not file_signature in captionsoverall:
                    self.logger.debug("File record not found in csv: {}".format(file))
                    continue
                cap_action_obj = captionsoverall[file_signature]
                cap_action_obj[0] = cap_action_obj[0].replace('put', 'push').replace('remove', 'pull').replace(
                    'take', 'pull')
                if cap_action_obj[0] not in self.action_set:
                    if self.consider_others and cap_action_obj[0] not in self.avoid_action_set:
                        cap_action_obj[0] = 'others'
                    else:
                        self.logger.debug("Ignoring action: {}".format(cap_action_obj[0]))
                        continue

                cap_count_total[cap_action_obj[0]] = cap_count_total.get(cap_action_obj[0], 0) + 1
                object_count_total[cap_action_obj[1]] = object_count_total.get(cap_action_obj[1], 0) + 1
                if cap_count_total[cap_action_obj[0]] > self.each_action:
                    # self.logger.debug("Action {} exceeds class limit".format(cap_action_obj[0]))
                    continue
                cap_count_used[cap_action_obj[0]] = cap_count_used.get(cap_action_obj[0], 0) + 1
                object_count_used[cap_action_obj[1]] = object_count_used.get(cap_action_obj[1], 0) + 1
                try:
                    features = pd.read_csv(os.path.join(feature_path, file), sep=' ', header=None,
                                           dtype=np.float32, engine='c', na_filter=False).as_matrix()
                    # print(type(features))
                    # print(features.shape)
                    # features = np.genfromtxt(os.path.join(feature_path, file), delimiter=" ", skip_header=True, skip_footer=True, dtype=np.float32)
                except:
                    self.logger.warning("Empty file: {}".format(file))
                if len(features) > self.max_video_length:
                    continue
                if len(features) > longest_sequence:
                    longest_sequence = len(features)
                data_count += 1
                object_vocab.add(cap_action_obj[1])
                if cap_action_obj[0] in self.action_set:
                    action_vocab.add(cap_action_obj[0])
                elif not cap_action_obj[0] in self.avoid_action_set:
                    action_vocab.add("others")
                cap_feat.append(self._padding_feat(features))
                cap_label.append(cap_action_obj)
        self.logger.info("Longest_sequence: {}".format(longest_sequence))
        self.logger.info("Total action hist: {}".format(cap_count_total))
        self.logger.info("Used action hist: {}".format(cap_count_used))
        self.logger.info("Total object hist: {}".format(object_count_total))
        self.logger.info("Used object hist: {}".format(object_count_used))
        self.logger.info("Done loading features")

        self.logger.info("Creating vocabulary dict")
        action_index = dict([(s[1], s[0] + capoffset) for s in enumerate(action_vocab)])
        object_index = dict([(s[1], s[0] + capoffset) for s in enumerate(object_vocab)])
        self.logger.info("Action vocab size: {}".format(len(action_index)))
        self.logger.info("Object vocab size: {}".format(len(object_index)))

        true_data_size = len(cap_feat)
        train_size = int(true_data_size * self.train_weight)
        test_size = int(true_data_size * self.test_weight)
        valid_size = int(true_data_size * self.valid_weight)
        self.logger.info("Train size: {}\nTest size: {}\nValid size: {}".format(train_size, test_size, valid_size))

        shuffleidx = utils.shuffle(true_data_size, self.config["random"]["seed"])

        def filter_dataset(start_index, end_index):
            # f = open("splits/cook" + str(start_index) + '_' + str(end_index) + ".txt", 'wt')
            # cap_infolist_set = []
            features = []
            labels = []
            features.append([cap_feat[i] for i in shuffleidx[start_index:end_index]])
            labels.append([[action_index[cap_label[i][0]], object_index[cap_label[i][1]]] for i in
                           shuffleidx[start_index:end_index]])
            return [np.delete(np.asarray(features, dtype=np.float32).squeeze(axis=0),
                              np.s_[-(self.max_video_length - longest_sequence):], axis=1),
                    np.asarray(labels, dtype=np.int16).squeeze(axis=0)]

        train_data = None
        valid_data = None
        test_data = None
        if self.load_train:
            train_data = filter_dataset(0, train_size)
        if self.load_test:
            test_data = filter_dataset(train_size, train_size + test_size)
        if self.load_valid:
            valid_data = filter_dataset(train_size + test_size, train_size + test_size + valid_size)

        self.feats["train"], self.labels["train"] = train_data
        self.feats["test"], self.labels["test"] = test_data
        self.feats["valid"], self.labels["valid"] = valid_data
        self.action_index, self.object_index, self.longest_sequence = action_index, object_index, longest_sequence
        self.action_size = len(action_index)
        self.object_size = len(object_index)

        with h5py.File(self.hdf5_file, "w") as f:
            f.create_dataset("train/feature", self.feats["train"].shape, dtype='f', data=self.feats["train"])
            f.create_dataset("train/label", self.labels["train"].shape, dtype='i', data=self.labels["train"])
            f.create_dataset("test/feature", self.feats["test"].shape, dtype='f', data=self.feats["test"])
            f.create_dataset("test/label", self.labels["test"].shape, dtype='i', data=self.labels["test"])
            f.create_dataset("valid/feature", self.feats["valid"].shape, dtype='f', data=self.feats["valid"])
            f.create_dataset("valid/label", self.labels["valid"].shape, dtype='i', data=self.labels["valid"])
        with open(self.pkl_file, "wb") as f:
            pkl.dump([self.action_index, self.action_size, self.object_index, self.object_size,
                      self.longest_sequence], f)

class BatchGenerator(object):
    def __init__(self, loader, batch_size):
        config = cfg.get_config()
        self.loader = loader
        self.offsets = dict(train=0, valid=0, test=0)
        self.batch_size = batch_size
        self.shuffled_idx = {}
        seed = config["random"]["seed"]
        if not self.loader.is_loaded:
            self.loader.load_data()
        self.shuffled_idx["train"] = utils.shuffle(loader.get_data_size("train"), seed=seed)
        self.shuffled_idx["test"] = utils.shuffle(loader.get_data_size("test"), seed=seed)
        self.shuffled_idx["valid"] = utils.shuffle(loader.get_data_size("valid"), seed=seed)

    def get_num_batches(self, subset):
        return self.loader.get_data_size(subset) // self.batch_size + 1

    def next_batch(self, subset, batch_size=None):
        """
        :param subset: one of train, test, valid
        :param batch_size: size of batch
        :return: a batch of data
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.offsets[subset] >= len(self.loader.feats[subset]):
            self.offsets[subset] = 0
            self.shuffled_idx[subset] = utils.shuffle(self.loader.get_data_size(subset))
            return None, None, None
        new_offset = self.offsets[subset] + batch_size
        feats = []
        actions = []
        objects = []
        i = self.offsets[subset]
        while i < min(new_offset, self.loader.labels[subset].shape[0]):
            ind = self.shuffled_idx[subset][i]
            # one_hot = np.zeros(self.loader.action_size)
            # one_hot[self.loader.labels[subset][ind][0]] = 1
            # actions.append(one_hot)
            actions.append(self.loader.labels[subset][ind][0])
            feats.append(self.loader.feats[subset][ind])
            # one_hot = np.zeros(self.loader.object_size)
            # one_hot[self.loader.labels[subset][ind][1]] = 1
            # objects.append(one_hot)
            objects.append(self.loader.labels[subset][ind][1])
            i += 1
        self.offsets[subset] = new_offset
        return np.asarray(feats), np.asarray(actions, dtype=np.int32), np.asarray(objects, dtype=np.int32)

    def get_epoch(self, subset):
        ipt, lb1, lb2 = self.next_batch(subset)
        while ipt is not None:
            yield ipt, lb1, lb2
            ipt, lb1, lb2 = self.next_batch(subset)

    def get_all(self, subset):
        feats, verbs, nouns = self.next_batch(subset, self.loader.get_data_size(subset))
        if feats is None:
            feats, verbs, nouns = self.next_batch(subset, self.loader.get_data_size(subset))
        self.next_batch(subset, 1)
        return feats, verbs, nouns

