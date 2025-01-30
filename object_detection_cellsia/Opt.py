
class Opt:
    def __init__(self, key):
        self.device = ''  # 'cuda:0' o 'cpu'
        self.project = 'runs/test'
        self.name = 'exp'
        self.exist_ok = False
        self.task = key 
        self.single_cls = False



