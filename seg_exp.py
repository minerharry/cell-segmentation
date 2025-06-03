#populated with all my own settings and such

from pathlib import Path
from typing import Literal

import ultraimport
from segment_movie import SegmentationParams, segment_movie


def segment_experiment(exp:str,
                       cell_model:str|None,nuc_model:str|None,
                       cell_model_type:Literal["keras","fastai","auto"]|None="auto",
                       nuc_model_type:Literal["keras","fastai","auto"]|None="auto",
                       split:bool=True,
                       cell_stack:bool=True,
                       nuc_stack:bool=True,
                       out_subfolder:str="segmentation_masks_out",
                       do_zip:bool=True,
                       bucket_phase:bool=False,
                       batch_size:int=32,
                       parallel_improc=False,
                       local_folder:str|Path|None=None,
                       gcp_transfer:str|Path|None="gcp_transfer", #share transfers
                       models_folder:str|Path="gs://optotaxisbucket/models",
                       auto_sublocal:bool=True):
    exp_im_path = f"gs://optotaxisbucket/movies/{exp}/{exp}" if not bucket_phase else f"gs://optotaxisbucket/movies/{exp}/Phase"
    exp_out_path = f"gs://optotaxisbucket/movie_segmentation/{exp}/" + out_subfolder
    
    if cell_model:
        cell_params = SegmentationParams(cell_stack,do_splitting=split,stack_duplicate_missing=True); #duplicate missing because segmentation
    else:
        cell_params = None
    if nuc_model:
        nuc_params = SegmentationParams(nuc_stack,do_splitting=split,stack_duplicate_missing=True);
    else:
        nuc_params = None

    if cell_model_type == "auto":
        if cell_model is None:
            cell_model_type = None
        elif cell_model.startswith("keras"):
            cell_model_type = "keras"
        elif cell_model.startswith("fastai"):
            cell_model_type = "fastai"
        else:
            raise ValueError(cell_model_type)
        
    if nuc_model_type == "auto":
        if nuc_model is None:
            nuc_model_type = None
        elif nuc_model.startswith("keras"):
            nuc_model_type = "keras"
        elif nuc_model.startswith("fastai"):
            nuc_model_type = "fastai"
        else:
            raise ValueError(nuc_model_type)

    return segment_movie(cell_model,nuc_model,
                         exp_im_path,exp_out_path,
                         cell_params,nuc_params,
                         cell_model_type=cell_model_type,
                         nuc_model_type=nuc_model_type,
                         do_zip=do_zip,
                         batch_size=batch_size,
                         models_folder=models_folder,
                         local_folder=local_folder,
                         gcp_transfer=gcp_transfer,
                         gpu=True,
                         parallel_improc=parallel_improc,
                         clear_on_close=False,
                         auto_sublocal=auto_sublocal)

if __name__ == "__main__":
    # from ..SegmenterProcessing.fetch_images import keydict
    # keydict = ultraimport("../SegmenterProcessing/fetch_images.py").keydict
    # exp = keydict["itsn43"]
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Segment Experiments',)
    parser.add_argument('-N','--nucleus_only',"--nuc_only")
    parser.add_argument('-C','--cell_only')
    parser.add_argument("experiments",nargs="*")
    
    args = parser.parse_args()

    exps = args.experiments
    nuc_only = args.nucleus_only
    cell_only = args.cell_only

    # exps = ["2023.2.1 OptoITSN Test 43","2024.6.27 OptoPLC FN+Peg Test 4"]
# 
    for exp in exps:


        print(f"segmenting experiment: {exp}")
        segment_experiment(exp,"keras/iter4_1_3stack_cell" if not nuc_only else None,"keras/iter5_1_3stack_nuc" if not cell_only else None,
                           out_subfolder="3stack_masks_out",local_folder=f"local/{exp}/cell",auto_sublocal=False)
    # segment_experiment(exp,None,"fastai/iter3_4_nuc_continue.pkl",out_subfolder="3stack_masks_out",local_folder=r"_local\3067739830736")