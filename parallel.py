
from typing import Iterable, List, Union, Callable, Generic, TypeVar
from ray.types import ObjectRef

from image_processing import Img, proc_type
R = TypeVar("R")
class ObjectRefGenerator(Generic[R]):
    """A generator to obtain object references
    from a task in a streaming manner.

    The class is compatible with generator and
    async generator interface.

    The class is not thread-safe.

    Do not initialize the class and create an instance directly.
    The instance should be created by `.remote`.

    >>> gen = generator_task.remote()
    >>> next(gen)
    >>> await gen.__anext__()
    """

    """
    Public APIs
    """

    def __iter__(self) -> "ObjectRefGenerator[R]": ...

    def __next__(self) -> ObjectRef[R]: ...

    def send(self, value): ...

    def throw(self, value): ...

    def close(self): ...

    def __aiter__(self) -> "ObjectRefGenerator[R]": ...

    async def __anext__(self): ...

    async def asend(self, value): ...

    async def athrow(self, value): ...

    async def aclose(self): ...

    def completed(self) -> ObjectRef[R]: ...

    def next_ready(self) -> bool: ...

    def is_finished(self) -> bool: ...




remote_proc_type = Callable[[Img|Iterable[Img]],ObjectRef[Img|Iterable[Img]]]
def remote_wrap(fn:proc_type,remote_kwargs={})->remote_proc_type:
    par = ray.remote(fn,**remote_kwargs)
    def remoted(im:Union[Img,ObjectRef[Img]])->ObjectRef[Img|Iterable[Img]]:
        return par.remote(im)
    setattr(remoted,"is_remoted",True)
    return remoted


def parallel_compose_proc_functions(funs:List[proc_type|remote_proc_type|tuple[proc_type,dict]])->proc_type:
    """ Parallel version of compose_proc_functions. NOTE: At every branch of the function tree 
    (each input image, functions that return multiple images) there is no guarantee that proc functions 
    will be run in the same thread. THEREFORE, ANY STATEFUL PROCESS FUNCTIONS MUST BE MADE PROCESS-SAFE WITH RAY ACTORS 
    (See Enumerator for an example).
    """

    raise NotImplemented
    #OK SO, STATE OF THIS
    #PROBLEM: EACH GENERATED TASK CAN PRODUCE A GENERATOR OF IMAGES - Looking at you, Enumerator

    if not IS_PARALLEL:
        raise ValueError("Cannot use parallel processing functions without activating a parallel context. Please use:\n" + \
        "with parallelize_actors():\n" + \
        "    [create process functions here]")
    remote_funs:list[remote_proc_type] = []
    for f in funs:
        if isinstance(f,tuple):
            remote_funs.append(remote_wrap(*f))
        elif getattr(f,"is_remoted",False):
            remote_funs.append(f) #already wrapped
        else:
            remote_funs.append(remote_wrap(f))
    
    # @ray.remote
    # def fetch_loop(images:ObjectRef[Img],func:remote_proc_type,out:ray.util.queue.Queue):
    #     for im in images:
    #         out.put(func(im))
    #     out.put(None)
    
    # @ray.remote
    # def push_loop(inp:ray.util.queue.Queue):

        
        

    @ray.remote
    class RayWaIterator(Iterator[ObjectRef[Img]]):
        def __init__(self,images:ObjectRefGenerator[Img],func:remote_proc_type):
            self.images = images
            self.func = func
            self.refs = []
            
        def iter(self): #this seems... silly, but I think we've got it
            yield from self

        def __iter__(self):
            return self

        def __next__(self):
            

            
            if len(self.refs) == 0:
                raise StopIteration
            ready:ObjectRefGenerator[Img]
            # print([type(ref) for ref in self.refs])
            [ready],self.refs = ray.wait(list(self.refs),fetch_local = True,num_returns=1)
            # print(type(ready))
            return next(ready);

    def exec(im:Union[Img,Iterable[Img]]):
        if isinstance(im,Img):
            im = [im]
        refs = im
        assert len(remote_funs) > 0
        for func in remote_funs:
           
           refs = func(refs)
        return RayWaIterator(refs)
    return exec



def print_res(fn,name):
    def f(*args,**kwargs):
        r = fn(*args,**kwargs)
        print(f"{name}: {r}")
        return r
    return f




import ray
import ray.actor
from builtins import dir
import types
### Using ray backend instead of loky for joblib, so that we can have shared classes (actors).
### However, ray.remote() on a class makes every function and attribute access **doubly** indirect,
### first by adding a .remote() to actually call it and returning an ObjectRef, then when calling ray.get() on that ObjectRef
### This is great if you want to have lots of async stuff or specify options when the task itself is completed,
### but the goal here is solely performance with limited interaction with Ray, so usually the .remote and the .get
### are totally useless! So this is a special wrapper that kind of un-does all the spcial methods ray adds by re-exposing them
class WrappedActor: #insert into the instance tree to detect wrapped actors
    pass
@doublewrap
def ray_remote_invisible(baseclass:type,/,attr_access=True,**kwargs):
    """takes the same arguments as ray.remote"""
    if issubclass(baseclass,WrappedActor):
        raise ValueError(f"Cannot wrapped already-wrapped actor class {baseclass}!");
    if attr_access:
        if hasattr(baseclass,"_get_actor_attr"):
            raise NameError(f"Name collision: cannot add actor attribute access to class {baseclass} with existing attribute _get_actor_attr.")
        def _get_actor_attr(self,name:str):
            # print(f"getting actor attribute: {name}")
            return getattr(self,name)
        baseclass._get_actor_attr = _get_actor_attr

    if kwargs:
        actor = ray.remote(**kwargs)(baseclass)
    else:
        actor = ray.remote(baseclass)

    class ActorWrapper(baseclass,WrappedActor):
        _actorbase = actor
        def __init__(self,*args,**kwargs):
            self._handle = self._actorbase.remote(*args,**kwargs)

            ##since the handle methods are created at instantiation, we have to create ours at instantiation, too
            for name in dir(self._handle):
                try:
                    v = getattr(self._handle,name)
                except:
                    continue
                # print(name,type(v))
                if isinstance(v,ray.actor.ActorMethod):
                    # print("adding method:",name)
                    ##since we are adding a method at instantiation, we have to bind it ourselves
                    setattr(self,name,types.MethodType(unremote_method(v),self))
            print("wrapper instantiation complete")


        def __getstate__(self):
            # print("Wrapper state called")
            return self.__dict__

        def __getattr__(self,name:str):
            """will only be called for attributes of the actor class that were not copied into the base actor as ActorMethods.
            The returned objects will be serialized copies, and not maintain their link with the original actor."""
            # print(f"getting attribute {name} for wrapper object {self}")
            if name == "__setstate__":
                raise AttributeError(name)
            if hasattr(self,"_get_actor_attr"):
                return self._get_actor_attr(name)
            else:
                raise Exception("Cannot use __getattr__; actor class does not implement _get_actor_attr")

    return ActorWrapper



def unremote_method(method:ray.actor.ActorMethod):
    def call(self,*args,**kwargs): #self input because method
        # print(f"remote method {method.remote.__name__} called")
        ref = method.remote(*args,**kwargs)
        # raise ValueError(f"Error while calling remote method {method.__name__} with args {args,kwargs}, returned ref {ref}")
        if isinstance(ref,ray.ObjectRefGenerator):
            # raise ValueError(ref)
            return (ray.get(r) for r in ref)
        elif isinstance(ref,ray.ObjectRef):
            return ray.get(ref)
        else:
            raise ValueError(f"Calling remote method {method} returned non-object-ref {ref}")

    call.__name__ = method.remote.__name__
    return call
