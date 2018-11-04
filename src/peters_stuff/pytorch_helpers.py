


_DEFAULT_DEVICE = 'cpu'


def set_default_device(device):
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def get_default_device():
    return _DEFAULT_DEVICE

