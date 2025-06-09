import os
from pathlib import Path
from typing import Any, Protocol
from imageio.v3 import imread
import numpy as np
import torch

# fix for new segmentation_models error was taken from here: https://stackoverflow.com/questions/75433717/module-keras-utils-generic-utils-has-no-attribute-get-custom-objects-when-im
os.environ["KERAS_BACKEND"] = "torch"
os.environ["SM_FRAMEWORK"] = "keras"

from keras.preprocessing.image import load_img
import keras


try:
    raise ImportError
    from tensorflow import Tensor as tfTensor
except:
    class _tfTensor:
        def numpy(self)->np.ndarray:
            raise NotImplemented
    tfTensor = _tfTensor
try:
    from torch import Tensor as torchTensor
except:
    class _torchTensor:
        def numpy(self)->np.ndarray:
            raise NotImplemented
    torchTensor = _torchTensor

import math
class TrainingSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size:int, img_size:tuple[int,int], input_img_paths:list[str], target_img_paths:list[str]):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

        assert len(input_img_paths) == len(target_img_paths)

    def __len__(self):
        return math.ceil(len(self.target_img_paths) / self.batch_size)


    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        if idx > len(self):
            raise StopIteration
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        if len(batch_input_img_paths) == 0 or len(batch_target_img_paths) == 0:
            raise StopIteration #sanity check
        x = np.zeros((len(batch_input_img_paths),) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode='rgb')
            x[j] = img
        y = np.zeros((len(batch_target_img_paths),) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y


class SegmentationSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    def __init__(self, batch_size:int, img_size:tuple[int,int]|None, input_img_paths:list[str]):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths


    def __len__(self):
        # print(f"input paths: {len(self.input_img_paths)}, batch size: {self.batch_size}, total length: {math.ceil(len(self.input_img_paths) / self.batch_size)}")
        return math.ceil(len(self.input_img_paths) / self.batch_size)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        if idx > len(self):
            raise StopIteration
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        if len(batch_input_img_paths) == 0:
            raise StopIteration #sanity check
        if self.img_size is None:
            ##hack
            self.img_size = imread(batch_input_img_paths[0]).shape[:2]
        x = np.zeros((len(batch_input_img_paths),) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode='rgb')
            x[j] = img
        return (x,batch_input_img_paths)


### Model
class Model(Protocol): #very dumb and only works for segmentation
    def predict(self,x:np.ndarray,verbose:bool=True)->torch.Tensor: ...

class ModelLoader(Protocol):
    def __call__(self,modelpath:Path|str|os.PathLike[str],modeltype:str,gpu:bool=False,**kwargs)->Model: ...

class KerasModel(Model):
    def __init__(self,model:keras.Model,gpu:bool):
        self.model = model
        self.gpu = gpu

    def predict(self,x:np.ndarray,verbose:int=1)->torch.Tensor:
        t = torch.tensor(x,device='cuda' if self.gpu else 'cpu')
        preds = self.model(t,training=False)
        # from IPython import embed; embed()
        #TODO: streamed output of predict() - theoretically better to pass in the PyDataset and get some sort of iterator out of preict
        return keras.ops.argmax(preds,-1).cpu()

try:
    from fastai.learner import load_learner,Learner
    from fastai.data.load import DataLoader
except ImportError:
    class FastaiModel(Model): pass
    class FastaiWrapper(KerasModel): pass
else:
    class FastaiModel(Model):
        def __init__(self,model:Learner,gpu:bool):
            self.model = model
            self.gpu = gpu
            if self.gpu:
                self.model.dls.cuda() #set train and test dataloaders to be on the gpu

        def predict(self,x:np.ndarray,verbose:bool|int=True)->torch.Tensor:
            #this is so dumb
            # t = torch.tensor(x,dtype=torch.uint8,device='cuda' if self.gpu else 'cpu')
            x = x.astype('uint8')
            
            # preds = self.model.model(t)
            # from IPython import embed; embed()
            # return torch.argmax(preds,-1).cpu()

            if len(x.shape) == 4:
                bs = x.shape[0]
            else:
                bs = 1
                x = [x]
            dl = self.model.dls.test_dl(x,bs=bs,num_workers=0)
            _,_,decoded = self.model.get_preds(dl=dl,with_decoded=True)
            # # print(x.shape)
            # # from IPython import embed; embed()
            # import pdb; pdb.set_trace()
            # # DL = DataLoader(dataset=x,bs=bs,device='cuda' if self.gpu else 'cpu')
            # res = self.model.predict(x[0]);
            # # from IPython import embed; embed()
            # import pdb; pdb.set_trace()
            return decoded.cpu()
    
    from keras.layers import TorchModuleWrapper
    class FastaiWrapper(KerasModel): #you know what, let's try this instead
        def __init__(self,model:Learner,gpu:bool):
            # self.model = model
            torchmodel = model.model
            if gpu:
                torchmodel = torchmodel.to('cuda')
            super().__init__(TorchModuleWrapper(torchmodel),gpu)

        def predict(self,x,verbose:bool|int=True):
            s = np.moveaxis(x,-1,-3)
            return super().predict(s)



def load_model_keras(path,gpu=False,compile_args:dict[str,Any]={},**kwargs)->KerasModel:
    model:keras.Model = keras.saving.load_model(Path(path).with_suffix(".keras"));
    keras.backend.clear_session()
    model.compile(**compile_args)
    if gpu:
        try:
            model = model.cuda()
        except:
            raise Exception("Could not initialize keras model to cuda device. MAKE SURE KERAS BACKEND IS TORCH! Any keras imports BEFORE setting os.environ[\"KERAS_BACKEND\"] = \"torch\" will break the backend")
    return KerasModel(model,gpu=gpu)

def load_model_fastai_torch(path,gpu=False,**kwargs)->Model:
    import fastai
    def label_func(x): return None; ##dummy function to make unpickling work, never used
    def mask_from_image(x): return None; ##dummy function to make unpickling work, never used
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.PurePosixPath
    import __main__
    setattr(__main__,"label_func",label_func)
    setattr(__main__,"mask_from_image",mask_from_image)
    try:
        learner:Learner = load_learner(path,cpu=not gpu)
    finally:
        pathlib.PosixPath = temp
    return FastaiModel(learner,gpu=gpu)

def load_model(modelpath:Path|str|os.PathLike[str],modeltype:str,gpu=False,**kwargs)->Model:
    d = {"keras":load_model_keras,"fastai":load_model_fastai_torch}
    if modeltype not in d:
        raise ValueError(f"Unsupported model type: {modeltype}. Please use custom model loader or use a type of either keras or fastai, the two builtin supported versions")
    return d[modeltype](modelpath,gpu=gpu,**kwargs);

