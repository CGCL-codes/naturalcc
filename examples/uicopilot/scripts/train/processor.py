from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore as TSemaphore
from threading import Thread
import multiprocessing
import os,sys
import traceback

class MulThreading:
    def __init__(self, max_workers=5, print_func=None):
        self.print=print_func
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semephore = TSemaphore(max_workers*3) # 一次性不能submit太多避免内存爆掉
 

    def add_task(self, task, args, cb):   
        self.semephore.acquire()     
        def handle_result(future):
            self.semephore.release()
            try:
                result = future.result()  # 如果在download函数中发生异常，它将在这里抛出                
                cb(result)
            except Exception:
                exception_info = traceback.format_exc()
                self.print('Error: {}'.format(exception_info))    
                    
        fu = self.executor.submit(task, *args)
        fu.add_done_callback(handle_result)

    def shutdown(self, force=False):
        self.executor.shutdown(wait= not force, cancel_futures= force)
        
class MultiProcessor:
    def __init__(self, name="", num_processes=int(multiprocessing.cpu_count()), print_func = print):
        self.pool = multiprocessing.Pool(processes=num_processes)
        self.semephore = multiprocessing.Manager().Semaphore(num_processes*2)
        self.name=name
        self.print= print_func

    def add_task(self, task, args, cb):
        self.semephore.acquire()
        self.pool.apply_async(self._task_wrap, (task ,self.print, *args), callback=self._callback(cb), error_callback=self._error_callback())
    
    def _error_callback(self):
        def error_callback(e):   
            self.print(f"{self.name}:{e}")    
        return error_callback
    
    @staticmethod
    def _task_wrap(task, print_func, *args, **kwargs):   
        try:
            return task(*args, **kwargs)        
        except Exception as e:
            # 捕获异常并附加堆栈信息      
            print_func(f"pid {os.getpid()}, {traceback.format_exc()}")       

    def _callback(self, original_callback):
        def callback(result):
            self.semephore.release()
            if original_callback is not None:
                original_callback(result)
        return callback
    
    def shutdown(self, force=False):
        if force:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()   