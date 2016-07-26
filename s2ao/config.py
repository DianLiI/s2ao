import os
import configparser


def gen_config(config):
    cwd = os.getcwd()
    # default values
    config.add_section("logging")
    config.set("logging", "level", "INFO")
    config.add_section("path")
    config.set("path", "working_dir", cwd)
    config.set("path", "data_root", os.path.join(cwd, 'data/small'))
    config.set("path", "data_root", os.path.join(cwd, 'data/small'))
    config.set("path", "hdf5_file", os.path.join(cwd, 'data/feature_lable.hdf5'))
    config.set("path", "pkl_file", os.path.join(cwd, 'data/parameter.pkl'))
    config.set("path", "log_dir", os.path.join(cwd, 'log'))

    config.add_section("data")
    config.set("data", "# The longest video sequence, default 25")
    config.set("data", "max_video_length", "25")
    config.set("data", "# feature_type: one of rgb, action, both")
    config.set("data", "feature_type", "both")
    config.set("data",
               '# total size: max size of data to use.\n# -1 means no limit but can still be constrained by each_action')
    config.set("data", "total_size", "12000")
    config.set("data",
               '# each_action, max number of each action class, try to make the classes balanced')
    config.set("data", "each_action", "2000")
    config.set("data", "# distribution of splits, sum is 1")
    config.set("data", "train_weight", "0.6")
    config.set("data", "valid_weight", "0.15")
    config.set("data", "test_weight", "0.25")
    config.set("data", "# actions to be included")
    config.set("data", "include_actions", "cut, stir")
    config.set("data", "# actions to be avoided")
    config.set("data", "avoid_actions", "chop, slice, spread")
    config.set("data", "# If set, all classes not in the above two lists are 'other'")
    config.set("data", "consider_others", "True")
    config.set("data", "# If set, only uses 10% of the data")
    config.set("data", "simple_test", "True")
    config.set("data", "phase", "train")

    config.add_section("random")
    config.set("random", "seed", "123")

    config.add_section("training")
    config.set("training", "batch_size", "100")
    config.set("training", "keep_prob", "0.5")
    config.set("training", "learning_rate", "0.001")
    config.set("training", "batch_size", "50")
    config.set("training", "hidden_size1", "1000")
    config.set("training", "hidden_size2", "1000")
    with open("s2ao_config.ini", mode="w") as f:
        config.write(f)
def get_config():
    config = configparser.ConfigParser(allow_no_value = True)
    config.optionxform = str
    if not config.read("s2ao_config.ini"):
        print("No config fie found, generating default settings.")
        gen_config(config)
    return config
