from torch.utils.data import DataLoader
from data.dataset3d import Dataset3d


def collate_3d(batch):
    """
        Make collate_3d empty is to generate batch_3d as a list of items from Dataset3d. Each item is a result from Dataset3d.__getitem__.
    :param batch:
    :return:
    """
    return batch


def create_loader_3d(config, phase):
    dataset3d = Dataset3d(config['dataset']['3d'], phase)
    #
    loader = DataLoader(dataset3d,
                        batch_size=config['loader']['3d'][phase]['batch_size'],
                        shuffle=config['loader']['3d'][phase]['shuffle'],
                        num_workers=config['loader']['3d'][phase]['num_workers'],
                        collate_fn=collate_3d)
    return loader
