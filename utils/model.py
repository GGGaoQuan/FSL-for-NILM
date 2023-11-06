from tqdm import tqdm

from protonets.utils import filter_opt
from protonets.models import get_model


def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)


def evaluate(model, data_loader, meters, desc=None):
    model.eval()  # 这里的model.eval()是为了在训练子任务的验证时，将模型设置为评估模式

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)  # few_shot.py中的loss函数
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
