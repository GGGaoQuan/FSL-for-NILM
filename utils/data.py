import protonets.data


def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds = protonets.data.omniglot.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds


def loadp(opt, splits):
    # if opt['data.dataset'] == 'plaid':  如果用这条语句还要改run_eval,run_train等脚本
    ds = protonets.data.plaid.load(opt, splits)
    return ds


def loadm(opt, splits):
    ds = protonets.data.miniImagenet.load(opt, splits)
    return ds


def loads(opt, splits):
    ds = protonets.data.svhn.load(opt, splits)
    return ds
