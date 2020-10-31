import yaml

def load_config_yaml(config_path):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    return config
