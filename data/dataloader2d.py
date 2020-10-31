from torch.utils.data import DataLoader
from data.dataset2d import Dataset2d


def create_loader_2d(batch, config, phase):
    dataset2d = Dataset2d(config['dataset']['2d'], batch, phase)
    loader = DataLoader(dataset2d,
                        batch_size=config['loader']['2d'][phase]['batch_size'],
                        shuffle=config['loader']['2d'][phase]['shuffle'],
                        num_workers=config['loader']['2d'][phase]['num_workers']
                        )
    """
        default collate function from Dataloader:
        For instance, if each data sample consists of a 3-channel image and an integral class label, i.e., each element of 
        the dataset returns a tuple (image, class_index), the default collate_fn collates a list of such tuples into a 
        single tuple of a batched image tensor and a batched class label Tensor. In particular, the default collate_fn has 
        the following properties:

        It always prepends a new dimension as the batch dimension.
        It automatically converts NumPy arrays and Python numerical values into PyTorch Tensors.
        It preserves the data structure, e.g., if each sample is a dictionary, it outputs a dictionary with the same set of 
        keys but batched Tensors as values (or lists if the values can not be converted into Tensors). Same for list s, 
        tuple s, namedtuple s, etc.
    """
    return loader
