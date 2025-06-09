### API for tracking, helper pipeline functions for scripting API. Extracted from original segmentation_training_refactor_keras.ipynb notebook
from abc import ABC
import datetime
import random
import types
import typing
import zipfile
from gsutilwrap import rsync, stat, copy
import pandas as pd
from warnings import warn
from skimage.io import imread, imsave, imshow
from skimage.exposure import rescale_intensity, adjust_gamma
from skimage.transform import resize
from typing import Callable,Dict, Generic, Iterator,List, Literal, NamedTuple, Protocol, Sequence,Tuple,Union,Any,Iterable, overload
import torch
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path, PurePath
import re
import shutil
import stat
import json_tricks
from libraries import filenames

from collections import OrderedDict
from copy import deepcopy
from pathlib import PurePosixPath






def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )

def cleardir(dir): #clears all files in dir without deleting dir
  print("dir cleared:",dir)
  for f in os.scandir(dir):
    if os.path.isdir(f): shutil.rmtree(f,onerror=on_rm_error); #just in case
    else: os.remove(f);

def rmdir(dir):
    shutil.rmtree(dir,onerror=on_rm_error)

def is_gcp_path(path:PurePath):
  return path.parts[0].lower() == "gs:";

def create_UNET():
    from segmentation_models import Unet
    return Unet('resnet34', encoder_weights='imagenet', classes=2, input_shape=(None,None,3), activation='softmax')

def _linux_cmd_zip(source,dest,recurse=False,compresslevel:Union[int,None]=None,bar=None,relative_to = ""):
    raise NotImplemented
    relative = Path(source)/relative_to
    dest = Path(dest)
    # if compresslevel is not None:
    #     compresstext = f"-{compresslevel}"
    #     if recurse:
    #         result = !cd "{relative}" && zip "{compresstext}" -r "{dest.name}" *
    #     else:
    #         result = !cd "{relative}" && zip "{compresstext}" "{dest.name}" *
    # else:
    #     if recurse:
    #         result = !cd "{relative}" && zip -r "{dest.name}" *
    #     else:
    #         result = !cd "{relative}" && zip "{dest.name}" *
    shutil.move(relative/dest.name,dest);
    return result

def _python_cmd_zip(source,dest,recurse=False,compresslevel:Union[int,None]=6,bar=False,relative_to = ""):
    if compresslevel is None:
        compresslevel = 6
    source = Path(source);
    relative = relative_to or source
    # relative = source/relative_to
    dest = Path(dest)
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
    paths = list((source.rglob("*") if recurse else source.iterdir()))
    if paths:
        paths = tqdm(paths, dynamic_ncols=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=compresslevel) as archive:
        for filepath in paths:
            # print("relative:",filepath.relative_to(relative))
            archive.write(filepath,arcname=filepath.relative_to(relative));
zipExists = shutil.which("zip")
cmd_zip = _linux_cmd_zip if zipExists else _python_cmd_zip
print("using python zip" if not zipExists else "using cmdline zip")

def _linux_cmd_unzip(source,dest,overwrite=False)->List[str]:
  raise NotImplemented
  flags = "-" + ("o" if overwrite else "")
#   if flags ==  "-":
#     result = !unzip "{source}" -d "{dest}" 
#   else:
#     result = !unzip "{flags}" "{source}" -d "{dest}" 
  return result

def _python_cmd_unzip(source,dest,overwrite=False):
    source = Path(source);
    dest = Path(dest);
    with zipfile.ZipFile(source,'r') as archive:
        for member in archive.infolist():
            file_path = dest/member.filename
            if not file_path.exists():
                archive.extract(member, dest)

unzipExists = shutil.which("unzip")
cmd_unzip = _linux_cmd_unzip if unzipExists else _python_cmd_unzip
print("using python unzip" if not unzipExists else "using cmdline unzip")

class Img(NamedTuple):
    name:str
    img:np.ndarray
    metadata:dict[str,Any]

    # def __init__(self,name:str,img:np.ndarray,metadata:dict[str,Any]):
    #     self.name = name
    #     self.img = img
    #     self.metadata = metadata

    def copy(self):
        return Img(self.name,self.img,self.metadata)

    def asdict(self):
        return {"name":self.name,"img":self.img,"metadata":self.metadata}

    def astuple(self):
        return (self.name,self.img,self.metadata)

    def __iter__(self):
        return iter(self.astuple())

    def __str__(self):
        return f"Image {self.name} of shape {self.img.shape} and metadata {list(self.metadata.keys())}"

_single_proc = Callable[[Img],Union[Img,Iterable[Img]]]
# _multi_proc =
proc_type = Callable[[Union[Img,Iterable[Img]]],Union[Img,Iterable[Img]]]

from functools import wraps
def doublewrap(f):
    '''
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    '''
    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec

from typing import DefaultDict,Dict,TypeVar
K = TypeVar('K')
V = TypeVar('V')
def merge_data(*dicts:Dict[K,V],require_all_present=True,require_all_equal=True)->Dict[K,V]:
    if not require_all_present:
        if not require_all_equal:
            dd = DefaultDict(list)
            for d in dicts: # you can list as many input dicts as you want here
                for key, value in d.items():
                    dd[key].append(value)
            return {k:v[0] for k,v in dd.items()}
        else:
            dd = {}
            null = object()
            for d in dicts:
                for key,value in d.items():
                    if key in dd:
                        if dd[key] is not null and dd[key] != value:
                            dd[key] = null
                    else:
                        dd[key] = value
            return {k:v for k,v in dd.items() if v is not null}

    else:
        if not require_all_equal:
            dd = {k:[] for k in dicts[0].keys()}
            for d in dicts:
                bad = []
                for key in dd.keys():
                    if key not in d:
                        bad.append(key)
                    else:
                        dd[key].append(d[key])
                [dd.pop(k) for k in bad]
            return {k:v[0] for k,v in dd.items()}
        else:
            dd = dicts[0].copy()
            for d in dicts:
                bad = []
                for key in dd.keys():
                    if key not in d:
                        bad.append(key)
                    else:
                        if dd[key] != d[key]:
                            bad.append(key)
                # print(bad)
                [dd.pop(k) for k in bad]
            return dd


#for both processing and deprocessing images
#this allows functions like splitting or stitching of images; splititng returns multiple images, stitching can return [] for all images concatenated into a larger one
@doublewrap
def proc_fn(fn:Union[_single_proc,proc_type],*,multi=False)->proc_type:
    if multi:
        assert isinstance(fn,proc_type) #type:ignore
    def new_proc(im:Union[Img,Iterable[Img]]):
        # print(im)
        if isinstance(im,Img) or multi:
            res = fn(im)
        else:
            r = (fn(i) for i in im)
            r = ([k] if isinstance(k,Img) else k for k in r)
            res = (x for l in r for x in l)
        # print(f"Result of applying function {fn}: {res}")
        return res
    return new_proc

@proc_fn
def no_op(i):
    return i

def _proc_wrap(func:proc_type):
    if getattr(func,"is_procwrap",False):
        if not typing.TYPE_CHECKING:
            return func #type: ignore
    def wrapped(im:Union[Img,Iterable[Img]])->Iterable[Img]:
        o = func(im);
        if (isinstance(o,Img)):
            o = [o];
        return o;
    setattr(wrapped,"is_procwrap",True)
    return wrapped;

# def _proc_wrap_generator(func:proc_type):
#     if getattr(func,"is_procwrap_generator",False):
#         if not typing.TYPE_CHECKING:
#             return func #type: ignore
#     def wrapped(im:Img)->Iterable[Img]:
#         o = func(im);
#         if (isinstance(o,Img)):
#             o = [o];
#         yield from o
#     setattr(wrapped,"is_procwrap_generator",True)
#     return wrapped;

def _proc_compose(f1:proc_type,f2:proc_type)->proc_type:
  wf1 = _proc_wrap(f1);
  wf2 = _proc_wrap(f2);
  def composed(im:Union[Img,Iterable[Img]]):
    return (k for l in (wf2(i) for i in wf1(im)) for k in l)
  return composed;

def compose_proc_functions(funcs:List[proc_type]):
  return functools.reduce(_proc_compose,funcs);


from typing import DefaultDict, cast
class Enumerator: #simple in-out mapper for splitting/stitching like operations
    @overload
    @classmethod
    def create(cls,process_fn:proc_type,enum_in:bool=False,multi=False)->proc_type: ...
    
    @overload
    @classmethod
    def create(cls,process_fn:Callable[[Iterable[Img]],Img|Iterable[Img]],enum_in:bool=False,multi=True)->proc_type: ...
    
    @classmethod
    def create(cls,process_fn,enum_in:bool=False,multi=False)->proc_type:
        """creates an Enumerator class to process named images with a simple numbering scheme; returns processing function to give to process/deprocess images
        args:
        - process_fn: function that processes image (or ordered set of output images from another enumerator) and returns an image or list of images to be enumerated
        - enum_in (optional): whether process_fn inputs a single image or an enumerated list
        """
        enum = cls(process_fn,enum_in);
        return proc_fn(enum.process)

    def __init__(self,proc_fn:proc_type,enum_in:bool,delimiter:str='-',enumeration_name="enumeration"):
        self.name = enumeration_name
        self.proc = proc_fn;
        self.m_in = enum_in;
        self.delim = delimiter;
        self.queue:Dict[str,Union[Dict[int,Img],List[Union[Img,None]]]] = DefaultDict(dict);

    def process(self,im):
        # print("processing image:",im)
        assert isinstance(im,Img)
        name,ext = os.path.splitext(im.name);
        if (self.m_in):
            #accumulate images until we get a full set, then process them as a batch and release the result

            spl = name.rindex(self.delim)
            base = name[:spl];
            num = name[spl+len(self.delim):];
            final_im = num.endswith('f')
            num = int(num.rstrip("f"))


            if isinstance(self.queue[base],dict):
                #Check for metadata or final image to tell us how large this batch is
                if "enumeration" in im.metadata and self.name in im.metadata["enumeration"]:
                    #if metadata, use that and we're done
                    total:int = im.metadata["enumeration"][self.name]["total"]
                    assert num == im.metadata["enumeration"][self.name]["index"],f"{num} != {im.metadata['enumeration'][self.name]['index']}"
                    assert base == im.metadata["enumeration"][self.name]["basename"]
                    t:List[Img|None] = [None]*total
                    self.queue[base] = t
                elif final_im:
                    #otherwise, keep waiting for the final image
                    listed:List[Union[Img,None]] = [None for _ in range(num+1)];
                    q = self.queue[base]
                    assert isinstance(q,dict)
                    for n,im in q.items():
                        listed[n] = im;
                    self.queue[base] = listed

            self.queue[base][num] = im;
            im = im.copy()

            flush = self.flush_queue()

            ##yield all images from a generator of generators
            res = (k for x in (self._format_out(self.proc(t),n,ext) for n,t in flush) for k in x)
            yield from res
        else:
            #process image(s) in, enumerating the result and releasing as a batch
            if "enumeration" not in im.metadata:
                im.metadata["enumeration"] = OrderedDict()
            im.metadata["enumeration"][self.name] = {}
            yield from self._format_out(self.proc(im),im.name,ext)

    def _format_out(self,out:Union[Img,Iterable[Img]],name:str,ext:str):
        # print(f"formatting out:",out,"with name:",name)

        if (not isinstance(out,Img)):
            out = list(out)
            for i,im in enumerate(out):
                im.metadata["enumeration"][self.name]["index"] = i
                im.metadata["enumeration"][self.name]["total"] = len(out) #to ensure it can't be mistaken for an imagename; "/" is never allowed in a filename on any system
                im.metadata["enumeration"][self.name]["basename"] = name
                yield Img(f"{name}{self.delim}{i}" + ("f" if i == len(out)-1 else "") + ext,im.img,im.metadata)
        else:
            yield from [Img(name,out.img,out.metadata)];

    def flush_queue(self):
        complete = [];
        for name,tiles in self.queue.items():
            if isinstance(tiles,list) and all([t is not None for t in tiles]): #final found, all tiles full
                tiles = cast(List[Img],tiles)
                complete.append(name);
                # print("image complete:",name,tiles)
                yield (name,tiles);
        # print("complete:",complete)
        for n in complete:
            del self.queue[n];

from contextlib import AbstractContextManager, contextmanager

### ANY PARALLEL ACTOR CLASS NAMES GO HERE
actor_classes = ["Enumerator"]
_parallel_store = DefaultDict(type(None))
_local_store = {}
IS_PARALLEL = False

@contextmanager
def parallelize_actors():
    from parallel import ray_remote_invisible
    global IS_PARALLEL
    init_ray()
    try:
        if not IS_PARALLEL: #no nesting
            # print(Enumerator)
            for cls in actor_classes:
                _local_store[cls] = globals()[cls]
                globals()[cls] = _parallel_store[cls] or ray_remote_invisible(_local_store[cls])
            IS_PARALLEL = True
            # print(Enumerator)
            # from IPython import embed; embed()
        yield
    finally:
        if IS_PARALLEL:
            for cls in actor_classes:
                _parallel_store[cls] = globals()[cls]
                globals()[cls] = _local_store[cls]
        IS_PARALLEL = False




from PIL import Image

exts = Image.registered_extensions()
supported_extensions = {ex for ex, f in exts.items() if f in Image.OPEN}

def get_image_files(path, recurse=True, folders=None):
    "Get image files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=supported_extensions, recurse=recurse, folders=folders)

def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders = folders if folders else []
    extensions = {e.lower() for e in extensions} if extensions else set()
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return res

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res



@proc_fn
def prepare_image(im:Img):
    name,image,data = im
    image = rescale_intensity(image);
    if (image.dtype != "uint8"):
        image = (image/256).astype('uint8');
    if (len(image.shape) != 3 or image.shape[2] != 3):
        image = np.stack((image,image,image),axis=2);
    return Img(name,image,data)

@proc_fn
def prepare_mask(im:Img):
    name,mask,data = im
    #TODO: update for combined/3-stacked masks
    if (len(mask.shape) == 3):
        mask = mask[:,:,0];
    mask = mask.copy()
    mask[mask>0] = 1;
    mask[mask<=0] = 0;
    mask = mask.astype('uint8');
    return Img(name,mask,data)



    
def create_split_image_process_fn(x_slices,y_slices,context_bounds,crop)->proc_type:
    @proc_fn
    def split_image(obj:Img):
        name,im,data = obj
        M = (im.shape[0]-context_bounds[0]-context_bounds[2]-crop[0]-crop[2])/y_slices;
        N = (im.shape[1]-context_bounds[1]-context_bounds[3]-crop[1]-crop[3])/x_slices;

        if int(M) != M or int(N) != N:
            raise Exception(f"ERROR: Image with size {im.shape[:2]} cannot be sliced into {x_slices} columns and {y_slices} rows\nwith context bounds of {context_bounds}; {M} and {N} not integers");
        else:
            M = int(M)
            N = int(N)
            tiles = [Img(name,im[y-context_bounds[0]:y+M+context_bounds[2],x-context_bounds[1]:x+N+context_bounds[3]],deepcopy(data))
                    for y in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M)
                    for x in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];
            return tiles
    return Enumerator.create(split_image)

def create_stitch_image_process_fn(x_slices,y_slices,context_bounds,crop)->proc_type:
    def stitch_image(tiles:Iterable[Img]):
        # print(f"stitch image called with {len(tiles)} tiles")
        stitchMasks = []
        assert not isinstance(tiles,Img)
        for i,(_,m,_) in enumerate(tiles):
            if isinstance(m,tfTensor) or isinstance(m,torchTensor):
                m = m.numpy().astype('uint8')
            y = i // x_slices;
            x = i % x_slices;
            imBounds = [crop[0]+context_bounds[0] if y != 0 else 0,m.shape[0]-crop[2]-context_bounds[2] if y != y_slices-1 else m.shape[0],crop[1]+context_bounds[1] if x != 0 else 0 ,m.shape[1]-crop[3]-context_bounds[3] if x != x_slices - 1 else m.shape[1]];
            stitchMasks.append(m[imBounds[0]:imBounds[1],imBounds[2]:imBounds[3]]);
        stitched = np.concatenate([np.concatenate(stitchMasks[i*x_slices:(i+1)*x_slices],axis=1) for i in range(y_slices)]);
        return Img("",stitched,merge_data(*(t.metadata for t in tiles)))
    return Enumerator.create(stitch_image,enum_in=True);


import re,functools
format_regex = filenames.filename_regex
default_regex = filenames.filename_regex_anybasename

#fairly versatile offset function.
def make_offsets_regex_fn(parse_regex:str,format_regex:str,idx_key:int):
    def get_offsets_regex(im:str,offsets:Sequence[int])->list[str]:
        m = re.match(parse_regex,im)
        if not m:
            raise ValueError(f"Invalid regex or invalid filename: unable to parse filename {im} with regex {parse_regex}")
        
        grps = m.groups()

        idx_val = int(grps[idx_key])
        
        off_names = []
        for off in offsets:
            off_val = idx_val + off
            off_grps = grps[:idx_key] + (off_val,) + grps[idx_key+1:]
            off_names.append(format_regex.format(*off_grps))
        return off_names
    return get_offsets_regex

default_offsets_fn = make_offsets_regex_fn(default_regex,format_regex,idx_key=2);

def create_stack_adjacents(num_stack:int,duplicate_missing:bool=False,offsets_fn:Callable[[str,Sequence[int]],list[str]]|None=default_offsets_fn,custom_regex_key:tuple[str,str,int]|None=None):
    """
        custom_regex_key is a tuple of (parse_regex, format_regex, idx_key) where parse_regex is a regex with N capturing groups, where the [idx_key]'th group is an integer component that specifies
        the image's index. Adding or subtracting one to the [idx_key]'th group should produce the adjacent image. Then, format_regex.format(*groups) formatted with the parse_regex's capturing groups 
        (potentially with an offset [idx_key] group) should match either the original image matched or one offset by the appropriate amount.

        For example, if my files consisted of:
        a1.jpg
        a2.jpg
        a3.jpg
        a4.jpg
        ...
        b1.jpg
        b2.jpg


        My parse regex could be '([a-z])(\\d)\\.jpg' and my format regex would be '{}{}\\.jpg'. If a1 was adjacent to a2, then my idx_key would be 1.
    """
    
    if custom_regex_key is not None and (offsets_fn is None or offsets_fn is default_offsets_fn):
        offsets_fn = make_offsets_regex_fn(*custom_regex_key)
    if offsets_fn is None:
        offsets_fn = default_offsets_fn

    @proc_fn
    def stack_adjacents(image:Img): #should come before image preparation
        # keyword = image.metadata["keyword"]
        if num_stack != 3:
            if num_stack == 1:
                raise Exception("Image stacking turned on, but only one image stack specified. Please just disable image stacking.")
            else:
                raise NotImplementedError("Because of fastai's UNET shape, only triple stacking (or no stacking) is supported at the moment. This is fixable if the model is edited, which is not implemented yet")

        num_lower = int((num_stack-1)/2)
        num_upper = num_stack - num_lower - 1
        source = Path(image.metadata["sourcefolder"])
        offsets = range(-num_lower,num_upper+1)
        assert 0 in offsets

        offset_regexes = offsets_fn(image.name,offsets)

        names = []
        final = []
        for off,reg in zip(offsets,offset_regexes):
            if off == 0:
                im = image.img
            else:
                reg = re.compile(reg)
                # from IPython import embed; embed()
                g = list(filter(reg.match,os.listdir(source)));
                # print(g)
                if len(g) == 0:
                    if duplicate_missing:
                        im = image.img
                        names.append(None)
                    else:
                        tqdm.write(f"Skipping image {image.name} which is missing an image adjacency {reg}.")
                        return []
                else:
                    if len(g) > 1:
                        tqdm.write(f"Warning: Multiple images match regex [somehow]: {g}")
                    n = g[0]
                    names.append(n)
                    im = imread(source/n)
            final.append(im)

        f = np.stack(final,axis=2)
        meta = image.metadata.copy()
        meta["adjacency_names"] = names
        return Img(image.name,f,meta)
    return stack_adjacents

def unstack_adjacents(im:Img):
    raise NotImplementedError("adjacent image unstacking not implemented yet")


def create_scale_image_process_fn(scale_factor:float)->proc_type:
    from skimage.util import img_as_bool
    @proc_fn
    def scale_image(im:Img):
        new_shape = [int(scale_factor*t) for t in im.img.shape[:2]] + [*im.img.shape[2:]]
        r = resize(im.img,new_shape,preserve_range=True).astype(im.img.dtype)
        # r = np.rint(),out=np.zeros(new_shape,im.img.dtype),casting='unsafe')
        return Img(im.name,r,deepcopy(im.metadata));
    return scale_image;


def create_gamma_image_process_fn(gammas:List[float])->proc_type:
    @proc_fn
    def gammafy_image(obj:Img):
        # image,data = obj
        return [Img(obj.name,adjust_gamma(obj.img,gamma),deepcopy(obj.metadata)) for gamma in gammas];
    return gammafy_image;



def create_duplicate_process_fn(num_duplicates:int)->proc_type:
    @proc_fn
    def duplicate_image(obj:Img):
        return [obj.copy() for _ in range(num_duplicates)]
    return duplicate_image


from datetime import datetime
class SourceRecord:
    def __init__(self,file:str|Path,parent:"SourceProcessor",overwrite=False):
        self.file = Path(file)
        self.parent = parent
        self.load_data(overwrite=overwrite)

    def load_data(self,overwrite=False):
        if (self.file.exists() and not overwrite):
            data = json_tricks.load(str(self.file))
            self._log:list[str] = data["log"]
            self.keyword_status:dict[str,dict[Literal['processed','deprocessed'],bool]] = DefaultDict(lambda: {'processed':False,'deprocessed':False},**data["keyword_status"])
            self.processed_outfolders:dict[str,str] = data["processed_outfolders"]

        else:
            self._log = []
            self.keyword_status = DefaultDict(lambda: {'processed':False,'deprocessed':False});
            self.processed_outfolders:dict[str,str] = {}
            self.save_data()

    def log(self,*entries:str,timestamp=True):
        if timestamp:
            time = datetime.now()
            entries = tuple(f"{time}: {entr}" for entr in entries)
        self._log.extend(entries)
        self.save_data()

    def clear(self):
        os.remove(self.file)
    
    def get_log(self):
        return self._log
    
    def processed(self,keyword:str,processed:bool=True,to:str|None=None,log=True):
        self.keyword_status[keyword]['processed'] = processed
        if to: self.set_processed_location(keyword,to);
        if log: self.log(f"keyword {keyword} processing complete")
        self.save_data()

    def set_processed_location(self,keyword:str,loc:str):
        self.processed_outfolders[keyword] = loc
        self.save_data()

    def get_processed_location(self,keyword:str):
        return self.processed_outfolders[keyword]

    def get_processed(self,keyword:str):
        return self.keyword_status[keyword]["processed"]

    def deprocessed(self,keyword:str,deprocessed:bool=True,log=True):
        self.keyword_status[keyword]['deprocessed'] = deprocessed
        if log: self.log(f"keyword {keyword} deprocessing complete")
        self.save_data()

    def get_deprocessed(self,keyword:str):
        return self.keyword_status[keyword]["deprocessed"]

    @property
    def data(self):
        return {"log":self._log,"keyword_status":self.keyword_status,"processed_outfolders":self.processed_outfolders,"keyword_map":self.parent._keyword_map,"keyword_sources":self.parent._keyword_sources};

    def save_data(self):
        self.file.parent.mkdir(parents=True,exist_ok=True)
        json_tricks.dump(self.data,str(self.file),indent=1)


class SourceProcessor(AbstractContextManager):
    def __init__(self,
                 models:str|Path|None=None,
                 training_images:str|Path|None=None,
                 training_masks:str|Path|None=None,
                 segmentation_images:str|Path|None=None,
                 segmentation_masks:str|Path|tuple[str|Path,...]|None=None,
                 local:str|Path|None=None,gcp_transfer:str|Path|None=None,
                 clear_on_exit:bool=True,
                 record_events:bool=True,record_file:str|Path="source_record.json",
                 overwrite_record=False,
                 **extra_keywords:str | Path | tuple[str | Path, ...]):

        self.clear_on_exit = clear_on_exit

        self.local = Path(local or f"_local/{id(self)}")
        self.auto_local = (local is None) #if autogenerated, clear on delete
        if local is None:
            print("no local folder provided, using autogenerated:",self.local)
        self.temp_folder = self.local/"_temp"
        self.meta_folder = self.local/"_meta"
        # self.meta_folder = Path(r"_local\1748162773840\_meta")
        self.gcp_transfer= Path(gcp_transfer or self.local/"_gcp_transfer")
        self.auto_gcp = (gcp_transfer is None) #if autogenerated, clear on delete
        if gcp_transfer is None:
            print("no gcp transfer provided, using autogenerated:",self.gcp_transfer)

        self._keyword_map:Dict[str,Tuple[str|PurePath|Tuple[str|PurePath,...],Path]] = {}
        self._keyword_sources:Dict[str,Union[None,PurePath,Tuple[PurePath,...]]] = DefaultDict(type(None))

        self.set_models(models)
        self.set_training_images(training_images)
        self.set_training_masks(training_masks)
        self.set_segmentation_images(segmentation_images)
        self.set_segmentation_masks(segmentation_masks)
        [self.set_keyword(key,word) for key,word in extra_keywords.items()]

        if record_events:
            self.record = SourceRecord(self.local/record_file,self,overwrite=overwrite_record)
        else:
            self.record = None


    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None, /):
        if exc_value:
            self.log_error(exc_type,exc_value,traceback)
        elif self.clear_on_exit:
            assert self.record is not None
            self.record.clear()
            self.clear_folders()

    def log_error(self,exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None, /):
        import traceback as tb
        err = "\n".join(tb.format_exception(exc_type,value=exc_value,tb=traceback));
        if self.record:
            self.record.log(err);

    def create(self): 
        #this is sort of a formality to require the use of the context manager. Makes all the local folders it knows of.
        #if set_keyword is called after creation, you'll need to call create() again.
        if not self.temp_folder.exists(): self.temp_folder.mkdir(parents=True,exist_ok=True)
        if not self.meta_folder.exists(): self.meta_folder.mkdir(parents=True,exist_ok=True)
        if not self.gcp_transfer.exists(): self.gcp_transfer.mkdir(parents=True,exist_ok=True)
        if not self.local.exists(): self.local.mkdir(parents=True,exist_ok=True)
        for key in self._keyword_map:
            m = self._keyword_map[key][1]
            if not m.exists():
                m.mkdir(parents=True,exist_ok=True)


    def clear_folders(self):
        if self.temp_folder.exists(): rmdir(self.temp_folder)
        if self.meta_folder.exists(): rmdir(self.meta_folder)
        if self.auto_gcp and self.gcp_transfer.exists(): rmdir(self.gcp_transfer)
        if self.auto_local and self.local.exists(): rmdir(self.local)
        for key in self._keyword_map:
            #only remove local folders, NOT SOURCEFOLDERS. 
            #This means we don't touch _keyword_map[0] (sources), and we don't touch
            #_keyword_sources - even though _keyword_sources *could* reference temp files, if they do, they should still
            #only be cleared iff they are in temp (dealt with) or they are in gcp transfer AND we want to clear the transfer.
            #Thus, any temporary keyword_sources will be dealt with on their own
            m = self._keyword_map[key][1] 
            if m.exists(): rmdir(m)



    def set_keyword(self,keyword:str,folder:str|Path|None|tuple[str|Path,...]):
        if folder is not None:
            m = self.local/keyword
            self._keyword_map[keyword] = (folder,m);
        elif keyword in self._keyword_map:
            del self._keyword_map[keyword]

    def get_local(self,keyword:str):
        m = self._keyword_map[keyword][1]
        return m
    
    def set_models(self,folder:str|Path|None):
        self.set_keyword("models",folder)
    
    def set_training_images(self,folder:str|Path|None):
        self.set_keyword("training_images",folder)

    def set_training_masks(self,folder:str|Path|None):
        self.set_keyword("training_masks",folder)

    def set_segmentation_images(self,folder:str|Path|None):
        self.set_keyword("segmentation_images",folder)

    def set_segmentation_masks(self,folder:str|Path|tuple[str|Path,...]|None):
        self.set_keyword("segmentation_masks",folder)
    

    def _fetch_gcp_files(self,in_path:PurePath,keyword:str,overwrite:bool)->Path: ##should not be called outside of other helper functions
        is_file = str(in_path).lower().endswith(('.zip','.tif','.tiff'));
        destination = self.gcp_transfer/keyword/(in_path.stem if is_file else in_path.name);
        if not(os.path.exists(destination)):
            os.makedirs(destination);
        # copy_out = Path(destination)/in_path.name;
        command_output = None;
        if (is_gcp_path(in_path)): #first part is gs:, must be removed to work with gcloud
            in_path = PurePosixPath(*in_path.parts[1:])
        if overwrite or len(os.listdir(destination)) == 0:
            try:
                if is_file:
                    copy(f"gs://{in_path}",destination,multithreaded=True,recursive=True)
                else:
                    rsync(f"gs://{in_path}",destination,multithreaded=True,recursive=True)
            except Exception as e:
                raise RuntimeError(f"Error while downloading {keyword} from bucket") from e;    
        if len(os.listdir(destination)) == 0:
            raise RuntimeError("Error: downloading failed for an unknown reason, 0 files in dest directory; Command Output:",command_output);
        if (not is_file):
            return destination; #we're done
        else:
            destination = destination/in_path.name;
        if (in_path.suffix == '.zip'):
            out_path = destination.with_suffix('');
            command_output = None;
            if (overwrite or not os.path.exists(out_path)):
                try:
                    cmd_unzip(destination,destination.parent,overwrite=True)
                except Exception as e:
                    raise RuntimeError(f"Error while unzipping {keyword}") from e;
            if not os.path.exists(out_path):
                raise RuntimeError(f"Error while unzipping (from GCP): zip file {destination.name} does not contain folder {destination.with_suffix('').name}");
            return out_path;
        elif (in_path.suffix.lower().startswith('.tif')):
            raise NotImplementedError("unstacking TIF files not yet supported");
        else:
            raise NameError("Invalid input suffix, input validation should have caught this >:(");

    def _push_gcp_files(self,in_folder:Path,out_path:PurePath,keyword:str,overwrite:bool):
        print(f"pushing gcp files from {in_folder} to {out_path}")
        assert os.path.isdir(in_folder);
        if (is_gcp_path(out_path)): #first part is gs:, must be removed to work with gcloud
            out_path = PurePosixPath(*out_path.parts[1:])

        if (out_path.suffix != ''): #file
            in_file = None;
            if (out_path.suffix.lower() == '.zip'):
                in_file = self.gcp_transfer/keyword/"zip"/out_path.name
                try:
                    cmd_zip(in_folder,in_file,relative_to=str(in_folder.parent))
                except Exception as e:
                    raise RuntimeError(f"Error while zipping {keyword} folder {in_folder} to zip file") from e;
            elif (out_path.suffix.lower().startswith('.tif')):
                raise NotImplementedError("stacking TIF files not yet supported");
            else:
                raise NameError("Invalid output suffix; can only zip or stack tiffs. If the output is a directory, please do not add a file suffix.")

            try:
                copy(in_file,f"gs://{out_path}",multithreaded=True,recursive=True,no_clobber=not overwrite);
            except Exception as e:
                raise RuntimeError(f"Error while uploading {keyword} to bucket") from e;
        else: #upload entire directory
            try:
                rsync(in_folder,f"gs://{out_path}",multithreaded=True,recursive=True)
            except Exception as e:
                raise RuntimeError(f"Error while uploading {keyword} to bucket") from e;
        return

    def _fetch_sourcefolder(self,keyword:str,overwrite=True)->Path:
        """returns the folder with the raw files, after unzip and or download if necessary."""
        print("fetching sourcefolder:",keyword)
        if (not overwrite and self._keyword_sources[keyword] is not None):
            res = self._keyword_sources[keyword]
            assert isinstance(res,Path) #incoming sourcefolders are singular; if cached, is a Path on this system
            return res;
        source = self._keyword_map[keyword][0];
        assert not isinstance(source,tuple) #incoming sourcefolders are singular
        source = PurePath(source)
        result = source;
        if (is_gcp_path(source)):
            result = self._fetch_gcp_files(source,keyword,overwrite=True);
        elif (source.suffix == '.zip'):

            destination = self.temp_folder/keyword;
            os.mkdir(destination);
            cmd_unzip(source,destination,overwrite=True)
            result = destination/source.stem;
            if (not os.path.exists(result)):
                raise RuntimeError(f"Error while unzipping: zip file {source} does not contain folder {source.stem}");
        if (self._keyword_sources[keyword] is not None and self._keyword_sources[keyword] != result):
            raise RuntimeError(f"Error fetching sourcefolder for keyword {keyword}: sourcefolder already exists in a different location - are you pushing before you fetch?");
        result = Path(result)
        self._keyword_sources[keyword] = result;
        return result;

    def _push_sourcefolder(self,keyword,overwrite=True):
        """inverse of fetch_sourcefolder - zips sourcefolder to original location or pushes to gcp if appropriate, otherwise noop"""
        print("pushing sourcefolder:",keyword)
        sourcefolders = self._keyword_sources[keyword];
        if (sourcefolders is None):
            raise RuntimeError("")
        if not isinstance(sourcefolders,tuple):
            sourcefolders = (sourcefolders,)
        dests = self._keyword_map[keyword][0];
        if not isinstance(dests,tuple):
            dests = (dests,)
        for sourcefolder,dest in zip(sourcefolders,dests):
            dest = PurePath(dest)
            assert isinstance(sourcefolder,Path) #since pushing, must be pushing from somewhere local!
            if (is_gcp_path(dest)):
                self._push_gcp_files(sourcefolder,dest,keyword,overwrite=overwrite);
            elif (dest.suffix == '.zip'):
                raise NotImplementedError("local sourcefolder zipping not implemented");
        pass;



    def _decopy_local(self,keyword:str,push_source=True): #copies files from local to dest, potentially through a temp sourcefolder (keyword_sources)
        if self._keyword_sources[keyword] is None:
            k = self._keyword_map[keyword][0]
            if not isinstance(k,tuple):
                self._keyword_sources[keyword] = self.temp_folder/keyword/PurePath(k).name
            else:
                self._keyword_sources[keyword] = tuple(self.temp_folder/keyword/PurePath(ki).name for ki in k)

        dests = self._keyword_sources[keyword]
        if not isinstance(dests,tuple):
            dests = (dests,)
        local = self._keyword_map[keyword][1];

        for dest in dests:
            assert isinstance(dest,Path) #should be local!
            print(f"{local}->{dest}")
            try:
                rsync(local,dest,multithreaded=True,recursive=True)
            except Exception as e:
                raise Exception("error copying",keyword,"files") from e;

        if push_source:
            self._push_sourcefolder(keyword);

    def deprocess_keyword(self,keyword:str,deprocess_fn:proc_type,push_source=True,do_warn=False,use_recorded=True): #like decopy, but instead of copying to the output, it instead *processes* files, *then* saves then to the output
        if use_recorded and self.record and self.record.get_deprocessed(keyword):
            print(f"keyword {keyword} already deprocessed")
            return
        _dests:Sequence[Path]|Path|None = self._keyword_sources[keyword]
        dests:Sequence[Path] = []
        if _dests is None:
            p_sources = self._keyword_map[keyword][0]; #p for prospective
            if not isinstance(p_sources,tuple):
                p_sources = (p_sources,)
            for p_source in p_sources:
                p_source = PurePath(p_source)
                k = p_source
                print("checking prospective source",p_source);
                if (is_gcp_path(p_source) or p_source.suffix != ''):
                    print("file/gcp found: requires temp folder")
                    dest = self.temp_folder/keyword/PurePath(k).stem
                else:
                    print("not file nor gcp: writing directly to source");
                    dest = Path(p_source)
                if not os.path.exists(dest):
                    os.makedirs(dest,exist_ok=True)
                dests.append(dest)
            self._keyword_sources[keyword] = tuple(dests)
        else:
            if isinstance(_dests,tuple):
                for d in _dests: #this is a little paranoid but I'm paranoid
                    assert isinstance(d,Path) #should be local!
                    dests.append(d)
            else:
                assert isinstance(_dests,Path)
                dests.append(_dests)

        local = self._keyword_map[keyword][1];

        unique_dests = list(set(dests)) #sometimes they go to the same temp folder

        for im in tqdm(os.listdir(local),desc="deprocessing " + keyword + "...", dynamic_ncols=True):
            path = local/im;
            if (path.suffix not in (".tif",".tiff",".TIF",".TIFF")):
                    if do_warn:
                            warn(f"non-image file in {keyword} dir: {im}");
                    continue;
            try:
                image = imread(path);
            except:
                raise RuntimeError("unable to read image",path);
            # print(image.shape)
            try:
                    data = json_tricks.load(str(self.meta_folder/keyword/im))
            except:
                    try:
                        data = json_tricks.load(str(self.meta_folder/"segmentation_images"/im))
                        data["keyword"] = keyword
                        data["sourcefolder"] = dests
                    except:
                        warn(f"Unable to read image processing metadata for image {im}")
                        data = {"keyword":keyword,"sourcefolder":dests}
            # tqdm.write(f"deprocessing image with shape: {image.shape}")
            processed = deprocess_fn(Img(im,image,data));
            # print(deprocess_fn)
            if isinstance(processed,Img):
                processed = [processed];
            # print(f"deprocessed: {processed}")
            for name,p,data_dict in processed:
                for dest in unique_dests:
                    # print(p.shape)
                    imsave(dest/name,p,check_contrast=False);

        if push_source:
            self._push_sourcefolder(keyword);
    
        if self.record:
            self.record.deprocessed(keyword)

    def _copy_local(self,keyword:str,overwrite_source=False):
        source = self._fetch_sourcefolder(keyword,overwrite=overwrite_source);
        dest = self._keyword_map[keyword][1];

        try:
            rsync(source,dest,multithreaded=True,recursive=True)
        except Exception as e:
            raise Exception("error copying",keyword,"files") from e;
        return dest;

    def process_keyword(self,keyword:str,process_fn:proc_type,overwrite_source=False,use_recorded=True,do_warn:bool=False):
        if use_recorded and self.record and self.record.get_processed(keyword):
            print(f"keyword {keyword} already processed")
            res = self.local/self.record.get_processed_location(keyword) #if the saved location (b) is absolute, a/b = b
            return res
        source = self._fetch_sourcefolder(keyword,overwrite=overwrite_source);
        dest:str|Path = self._keyword_map[keyword][1];
        if os.path.exists(dest) and any(os.scandir(dest)):
            cleardir(dest);

        for im in tqdm(os.listdir(source),desc="processing " + keyword + "...", dynamic_ncols=True):
            path = source/im;
            if (path.suffix not in (".tif",".tiff",".TIF",".TIFF")):
                    if do_warn:
                        warn(f"non-image file in {keyword} dir: {im}");
                    continue;
            image = imread(path);
            data_dict = {"sourcefolder":source,"keyword":keyword,"history":[]}
            processed = process_fn(Img(im,image,data_dict));
            if (isinstance(processed,Img)):
                processed = [processed];
            os.makedirs(self.meta_folder/keyword,exist_ok=True)
            for name,p,data in processed:
                # print(name)
                imsave(dest/name,p,check_contrast=False);
                json_tricks.dump(data,str(self.meta_folder/keyword/name),indent=1)

        if self.record:
            strdest = str(dest)
            if self.local in dest.parents:
                strdest = str(dest.relative_to(self.local))
            self.record.processed(keyword,to=strdest)

        return dest;

    def process_training_images(self,process_fn:proc_type=no_op,overwrite_source=False):
        return self.process_keyword("training_images",process_fn,overwrite_source=overwrite_source);

    def process_segmentation_images(self,process_fn:proc_type=no_op,overwrite_source=False):
        return self.process_keyword("segmentation_images",process_fn,overwrite_source=overwrite_source);

    def process_training_masks(self,process_fn:proc_type=no_op,overwrite_source=False):
            # breakpoint()
            return self.process_keyword("training_masks",process_fn,overwrite_source=overwrite_source);

    def deprocess_segmentation_masks(self,process_fn:proc_type=no_op):
        self.deprocess_keyword("segmentation_masks",process_fn,push_source=True);

    def fetch_modelsfolder(self,overwrite=False):
        return self._copy_local("models",overwrite_source=overwrite);

    def push_modelsfolder(self,fetch=False):
        if (fetch):
            self.fetch_modelsfolder(overwrite=False);
        return self._decopy_local("models",push_source=True);

    def push_segmentation_folder(self):
        return self._push_sourcefolder("segmentation_masks")


def get_shared_names(folders:Iterable[Path|str]):
    sharedpaths = set.intersection(*map(set,map(os.listdir,folders)))
    return sharedpaths

def get_shared_files(folders:Iterable[Path|str]):
    return [tuple(f/p for f in folders) for p in get_shared_names(folders)]


def init_ray():
    import ray
    if not ray.is_initialized():
        ray.init(dashboard_host="0.0.0.0") #allow external connections (e.g. over tailscale since no port-forwarding through firewall)
