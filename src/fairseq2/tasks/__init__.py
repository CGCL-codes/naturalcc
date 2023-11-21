from .ncc_task import NccTask

TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def register_task(name):
    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError()
        if not issubclass(cls,NccTask):
            raise ValueError()
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(name)
        return cls
        
    return register_task_cls