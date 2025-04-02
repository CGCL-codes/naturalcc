from .ncc_task import NccTask

TASK_REGISTRY = {}

def register_task(name):
    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError()
        if not issubclass(cls,NccTask):
            raise ValueError()
        TASK_REGISTRY[name] = cls
        return cls
        
    return register_task_cls