import contextlib
from dataclasses import dataclass
import itertools
import os
from pathlib import Path
import random
from typing import Callable, Iterable, Literal, Protocol, Sequence
from imageio import imread
from imageio.v3 import imwrite
import torch
from tqdm import tqdm

from image_processing import compose_proc_functions, create_scale_image_process_fn, create_split_image_process_fn, create_stack_adjacents, create_stitch_image_process_fn, get_image_files, get_shared_names, prepare_image, prepare_mask, proc_type, SourceProcessor, unstack_adjacents
from models import Model, ModelLoader, SegmentationSequence, TrainingSequence, load_model


### Training Stuff
def get_training_sequences(imfolder,maskfolder,train_percent = 0.8, batch_size:int=8):
    sharedpaths = [(imfolder/p,maskfolder/p) for p in get_shared_names([imfolder,maskfolder])]
    print(f"{len(sharedpaths)} shared file paths")
    im_size = imread(sharedpaths[0][0]).shape
    mask_size = imread(sharedpaths[0][1]).shape
    if im_size[:2] != mask_size[:2]:
        raise NotImplementedError(im_size,mask_size)

    random.shuffle(sharedpaths)

    print(f"creating training sequence with {len(sharedpaths)} (image,mask) pairs")
    
    train_samples = int(len(sharedpaths)*train_percent)
    train,val = sharedpaths[:train_samples],sharedpaths[train_samples:]

    train_sequence = TrainingSequence(batch_size,im_size[:2],[p[0] for p in train],[p[1] for p in train])
    val_sequence = TrainingSequence(batch_size,im_size[:2],[p[0] for p in val],[p[1] for p in val])

    return (train_sequence,val_sequence)

### Segmentation Stuff
def get_segmentation_sequence(input_folder:str|Path,*output_folders:str|Path,recurse=True,batch_size:int=32,overwrite=False):
    ## prepare input file reading
    files = get_image_files(input_folder,recurse=recurse)

    print(f"creating segmentation sequence with {len(files)} image files")
    #make destination folders to match input directory tree
    parents:set[Path] = set()
    for name in files:
        for d in output_folders:
            parents.add(Path(os.path.relpath(name,input_folder)).parent);
    
    for par in parents:
        for d in output_folders:
            (Path(d)/par).mkdir(parents=True,exist_ok=True);

    completed_masks = [set()];
    if not overwrite:
        completed_masks = []
        for d in output_folders:
            cmasks = set([os.path.relpath(x,d) for x in get_image_files(d,recurse=recurse)]); #optimize for lookup
            completed_masks.append(cmasks);

    files = [str(fi) for fi in files if any([os.path.relpath(fi,input_folder) not in cmasks for cmasks in completed_masks])];

    return SegmentationSequence(batch_size,None,files);


@dataclass
class SegmentationParams:
    stack_images:bool
    do_splitting:bool = False


    ##image stacking params
    num_stacked_images:int = 3
    mask_stacking_type:Literal["Narrow","Wide"] = "Narrow" #@param ["Narrow", "Wide"]
    stack_duplicate_missing:bool=False
    @property
    def mask_stacking(self):
        return (self.mask_stacking_type == "Wide")
    
    stacking_regex_key:None|tuple[str,str,int] = None
    


    ##down/upscaling params
    scale_factor:int = 1 #@param {type:"number"}
    @property
    def do_scaling(self):
        return self.scale_factor != 1



    ##image splitting/stitching params

    #Number of slices (columns/rows) to divide input images into; for the math, please see https://www.desmos.com/calculator/xrqev2vluo
    x_slices:int = 4
    y_slices:int = 5
    #dx, dy are the extra context around the segmented center in both directions
    dx:int = 32
    dy:int = 32
    #x and y crop are how much to straight remove from the image to make the sizes able to be subdivided nicely
    x_crop:int = 0
    y_crop:int = 0


    #both of these are negative y, negative x, positive y, positive x
    @property
    def crop(self):
        return [self.y_crop,self.x_crop]*2 
    @property
    def context_bounds(self):
        return [self.dy,self.dx]*2 #assuming x and y bounds are symmetrical to both sides of the image, which might not always true -- fix?



def make_process_fn(params:SegmentationParams,im_type:Literal["masks","images"],parallel=False)->proc_type:
    from image_processing import parallelize_actors
    if parallel:
        ctx = parallelize_actors
    else:
        ctx = contextlib.nullcontext
    with ctx():
        comp_fns = [];

        if params.stack_images and im_type == "images" or params.mask_stacking and im_type == "masks":
            comp_fns.append(create_stack_adjacents(params.num_stacked_images,duplicate_missing=params.stack_duplicate_missing,custom_regex_key=params.stacking_regex_key))

        if im_type == "images":
            comp_fns.append(prepare_image)
        else:
            comp_fns.append(prepare_mask)

        if params.do_splitting:
            comp_fns.append(create_split_image_process_fn(params.x_slices,params.y_slices,params.context_bounds,params.crop));

        if params.do_scaling:
            comp_fns.append(create_scale_image_process_fn(params.scale_factor))

        return compose_proc_functions(comp_fns) if not parallel else parallel_compose_proc_functions(comp_fns)

def make_deprocess_fn(params:SegmentationParams,im_type:Literal["masks","images"],parallel=False,)->proc_type:
    if parallel:
        ctx = parallelize_actors
    else:
        ctx = contextlib.nullcontext
    with ctx():
        comp_fns = [];

        if params.stack_images and im_type == "images" or params.mask_stacking and im_type == "masks":
            comp_fns.append(unstack_adjacents)

        if im_type == "images":
            comp_fns.append(prepare_image)
        else:
            comp_fns.append(prepare_mask)

        if params.do_splitting:
            comp_fns.append(create_stitch_image_process_fn(params.x_slices,params.y_slices,params.context_bounds,params.crop));

        if params.do_scaling:
            comp_fns.append(create_scale_image_process_fn(params.scale_factor))

        return compose_proc_functions(comp_fns) if not parallel else parallel_compose_proc_functions(comp_fns)


def segment_images(modelname:str,
                   im_source:str|Path,
                   mask_out:str|Path|tuple[str|Path,...], 
                   proc_fn:proc_type,
                   deproc_fn:proc_type,
                   batch_size:int=32,
                   modeltype:str="keras",
                   load_model:ModelLoader=load_model,
                   models_folder:str|Path="gs://optotaxisbucket/models",
                   local_folder:str|Path|None=None,
                   gcp_transfer:str|Path|None=None,
                   clear_on_close:bool=False,
                   gpu=True):

    with SourceProcessor(models=models_folder,
                             segmentation_images=im_source,
                             local=local_folder,
                             gcp_transfer=gcp_transfer,
                             segmentation_masks=mask_out,
                             clear_on_exit=clear_on_close) as source:

        print("processing images...")
        im_folder = source.process_segmentation_images(proc_fn)
        out_folder = Path(source.get_local("segmentation_masks"))
        print("image processing complete")
        
        print("segmenting to output folder:",out_folder)
        seg_sequence = get_segmentation_sequence(im_folder,out_folder,batch_size=batch_size)
        

        modelpath = source.fetch_modelsfolder()
        model = load_model(Path(modelpath)/modelname,modeltype,gpu=gpu,compile_args=dict(optimizer="rmsprop",loss=None,))

        print(f"predicting {len(seg_sequence)*batch_size} masks...")    
        for batch,inpaths in tqdm(seg_sequence,desc="predicting", dynamic_ncols=True):
            predictions:torch.Tensor = model.predict(batch,verbose=(len(batch) <= 1))
            print("preds_shape:",predictions.shape)
            for path,prediction in zip(inpaths,predictions):
                name = Path(path).relative_to(im_folder)
                im = prediction.numpy().astype('uint8');
                imwrite(out_folder/name,im);
        print("prediction complete")

        print("deprocessing masks...")
        source.deprocess_segmentation_masks(deproc_fn)
    print("mask deprocessing complete. Segmentation complete")


# for multiple models which takes images processed the same, saving some compute time
def multisegment_images(modelnames:list[str], 
                   im_source:str|Path,
                   mask_outs:Sequence[str|Path|tuple[str|Path,...]], 
                   proc_fn:proc_type,
                   deproc_fns:list[proc_type],
                   batch_size:int=32,
                   modeltypes:Iterable[str] = itertools.cycle(["keras"]),
                   load_model:ModelLoader=load_model,
                   models_folder:str|Path="gs://optotaxisbucket/models",
                   local_folder:str|Path|None=None,
                   gcp_transfer:str|Path|None=None,
                   clear_on_close:bool=False,
                   gpu=True):

    mask_keywords = [f"segmentation_masks_{i}" for i in range(len(mask_outs))]
    mask_sources = dict(zip(mask_keywords,mask_outs))

    with SourceProcessor(models=models_folder,
                             segmentation_images=im_source,
                             local=local_folder,
                             gcp_transfer=gcp_transfer,
                             clear_on_exit=clear_on_close,
                             **mask_sources) as source:

        print("processing images...")
        im_folder = source.process_segmentation_images(proc_fn)
        out_folders = [Path(source.get_local(mask)) for mask in mask_keywords]
        print("segmenting to output folder:",out_folders)
        seg_sequence = get_segmentation_sequence(im_folder,*out_folders,batch_size=batch_size)
        print("image processing complete")
        
        modelpath = source.fetch_modelsfolder()
        models:list[Model] = []
        for modelname,model_type in zip(modelnames,modeltypes):
            
            model = load_model(Path(modelpath)/modelname,model_type,gpu=gpu,compile_args=dict(optimizer="rmsprop",loss=None,))
            models.append(model)
            

        print(f"predicting {len(seg_sequence)*batch_size} masks...")    
        for batch,inpaths in tqdm(seg_sequence,leave=False, desc="predicting", dynamic_ncols=True):
            for model,out_folder in zip(models,out_folders):
                predictions = model.predict(batch,verbose=(len(batch) <= 1))
                for path,prediction in zip(inpaths,predictions):
                    name = Path(path).relative_to(im_folder)
                    im = prediction.numpy().astype('uint8');
                    imwrite(out_folder/name,im);
        print("prediction complete")

        print("deprocessing masks...")
        [source.deprocess_keyword(keyword,deproc) for keyword,deproc in zip(mask_keywords,deproc_fns)]
    print("mask deprocessing complete. Segmentation complete")


def segment_movie(cell_model:str|None,nuc_model:str|None,
                  in_path:str|Path,
                  out_folder:str|Path,
                  cell_params:SegmentationParams|None,
                  nuc_params:SegmentationParams|None,
                  do_rsync=True,
                  do_zip=True,
                  batch_size:int=32,
                  cell_model_type:str|None="keras",
                  nuc_model_type:str|None="keras",
                  load_model:ModelLoader=load_model,
                  models_folder:str|Path="gs://optotaxisbucket/models/keras",
                  parallel_improc=False,
                  local_folder:str|Path|None=None,
                  auto_sublocal:bool=True,
                  gcp_transfer:str|Path|None=None,
                  global_gcp:bool=True,
                  gpu=True,
                  clear_on_close:bool=True
                  ):
    if cell_model is None and nuc_model is None:
        return
    if gcp_transfer is None and global_gcp:
        if "gcp_transfer" in os.environ:
            gcp_transfer = os.environ["gcp_transfer"]
    out_folder = Path(out_folder)
    _outs:tuple[list[tuple[Path,...]],list[tuple[Path,...]]] = [(out_folder/"Cell",),(out_folder/"Nucleus",)] if do_rsync else [tuple(),tuple()],[(out_folder/"Cell.zip",),(out_folder/"Nucleus.zip",)] if do_zip else [tuple(),tuple()]
    outs = [(_outs[0][0] + _outs[1][0]),(_outs[0][1] + _outs[1][1])]

    if cell_params == nuc_params: #mutiple models with same in/out processing. Combine the image processing
        assert (cell_model is not None and nuc_model is not None and cell_params is not None and nuc_params is not None and cell_model_type is not None and nuc_model_type is not None)
        proc_fn = make_process_fn(cell_params,im_type='images',parallel=parallel_improc)
        deproc_fn = make_deprocess_fn(cell_params,im_type='masks',parallel=parallel_improc)
        multisegment_images([cell_model,nuc_model],
                            in_path,
                            outs,
                            proc_fn,
                            [deproc_fn,deproc_fn],
                            batch_size=batch_size,
                            models_folder=models_folder,
                            modeltypes=(cell_model_type,nuc_model_type),
                            load_model=load_model,
                            local_folder=local_folder,
                            gcp_transfer=gcp_transfer,
                            clear_on_close=clear_on_close,
                            gpu=gpu)
                            
    else:
        if local_folder is not None and auto_sublocal:
            cell_local = Path(local_folder)/"cell"
            nuc_local = Path(local_folder)/"nuc"
        else:
            cell_local = local_folder
            nuc_local = local_folder
        proc_fns = [make_process_fn(cell_params,im_type='images',parallel=parallel_improc) if cell_params is not None else None,make_process_fn(nuc_params,im_type='images',parallel=parallel_improc) if nuc_params is not None else None]
        deproc_fns = [make_deprocess_fn(cell_params,im_type='masks',parallel=parallel_improc) if cell_params is not None else None,make_deprocess_fn(nuc_params,im_type='masks',parallel=parallel_improc) if nuc_params is not None else None]
        if cell_model is not None:
            assert cell_model_type is not None
            p,d = proc_fns[0],deproc_fns[0]
            assert p is not None and d is not None
            segment_images(cell_model,
                        in_path,
                        outs[0],
                        p,
                        d,
                        batch_size=batch_size,
                        models_folder=models_folder,
                        modeltype=cell_model_type,
                        load_model=load_model,
                        local_folder=cell_local,
                        gcp_transfer=gcp_transfer,
                        clear_on_close=clear_on_close,
                        gpu=gpu
            )
        if nuc_model is not None:
            assert nuc_model_type is not None
            p,d = proc_fns[1],deproc_fns[1]
            assert p is not None and d is not None
            segment_images(nuc_model,
                        in_path,
                        outs[1],
                        p,
                        d,
                        batch_size=batch_size,
                        models_folder=models_folder,
                        modeltype=nuc_model_type,
                        load_model=load_model,
                        local_folder=nuc_local,
                        gcp_transfer=gcp_transfer,
                        clear_on_close=clear_on_close,
                        gpu=gpu
            )



# def segment_images_multi(models:list[str|Path],
#                   im_source:str|Path,
#                   mask_outs:list[str|Path],
#                   proc_fns:proc_type|list[proc_type],
#                   deproc_fns:list[proc_type],
#                   batch_size:int=32,
#                   models_folder:str|Path="gs://optotaxisbucket/models/keras",
#                   gpu=True):

#     source = SourceProcessor(models=models_folder,segmentation_images=im_source,segmentation_masks=mask_out)

#     im_folder = source.process_segmentation_images(proc_fn)
#     out_folder = Path(source.get_local("segmentation_masks"))
#     seg_sequence = get_segmentation_sequence(im_folder,out_folder,batch_size=batch_size)

#     model = load_model(Path(source.get_local("models"))/modelname,gpu=gpu)
#     model.compile(
#         optimizer="rmsprop",
#         loss=None,
#     )
    

#     for batch,inpaths in seg_sequence:
#         pred = model.predict(batch,verbose=(len(batch) <= 1))
#         predictions = keras.ops.argmax(pred,-1).cpu() #convert parallel confidences to label index
#         for path,prediction in zip(inpaths,predictions):
#             name = Path(path).relative_to(im_folder)
#             im = prediction.numpy().astype('uint8');
#             imsave(out_folder/name,im,check_contrast=False);

#     source.deprocess_segmentation_masks(deproc_fn)
