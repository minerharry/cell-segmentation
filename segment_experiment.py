#populated with all my own settings and such

from argparse import _ExtendAction, Action, ArgumentParser, FileType, Namespace
from ctypes import ArgumentError
import itertools
from typing import Tuple, TypeVar
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

from segmentation_and_training import SegmentationParams, segment_movie


def segment_experiment(in_path:str|Path,out_path:str|Path,
                       cell_model:str|None,nuc_model:str|None,
                       cell_model_type:Literal["keras","fastai","auto"]|None="auto",
                       nuc_model_type:Literal["keras","fastai","auto"]|None="auto",
                       split:bool=True,
                       cell_stack:bool=True,
                       nuc_stack:bool=True,
                    #    out_subfolder:str="segmentation_masks_out",
                       do_zip:bool=True,
                       bucket_phase:bool=False,
                       batch_size:int=32,
                       parallel_improc=False,
                       local_folder:str|Path|None=None,
                       gcp_transfer:str|Path|None="gcp_transfer", #share transfers
                       models_folder:str|Path="gs://optotaxisbucket/models",
                       auto_sublocal:bool=True):
    # exp_im_path = f"gs://optotaxisbucket/movies/{exp}/{exp}" if not bucket_phase else f"gs://optotaxisbucket/movies/{exp}/Phase"
    # exp_out_path = f"gs://optotaxisbucket/movie_segmentation/{exp}/" + out_subfolder
    
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
                         in_path,out_path,
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

T = TypeVar("T")
F = TypeVar("F")
def paired(iterable:Iterable[T])->Iterable[Tuple[T,T]]:
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    it = iter(iterable)
    return itertools.zip_longest(it,it)

_T = TypeVar("_T")
class PairedExtendAction(_ExtendAction):
    def __init__(self, option_strings: Sequence[str], dest: str, const: _T | None = None, default: _T | str | None = None, type: Callable[[str], _T] | FileType | None = None, choices: Iterable[_T] | None = None, required: bool = False, help: str | None = None, metavar: str | tuple[str, ...] | None = None) -> None:
        nargs = "+"
        super().__init__(option_strings, dest, nargs, const, default, type, choices, required, help, metavar)

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Sequence[Any] | None, option_string: str | None = None) -> None:
        if values is None or isinstance(values,str):
            raise ArgumentError();
        if len(values) % 2 != 0:
            raise ArgumentError("Must provide pairs of {in_folder} {out_folder} pairs")
    
        pairs = list(paired(values))
        return super().__call__(parser, namespace, pairs, option_string)


if __name__ == "__main__":

    ### command line input
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Segment Experiments',)
    parser.add_argument('-N','--nucleus_only',"--nuc_only")
    parser.add_argument('-C','--cell_only')
    parser.add_argument("experiment_pairs",action=PairedExtendAction)
    
    args = parser.parse_args()

    exps:list[tuple[str,str]] = args.experiment_pairs
    nuc_only = args.nucleus_only
    cell_only = args.cell_only

    print("Segmenting experiments:",exps)


    #models:
    # 3stack cell - "iter4_1_3stack_cell"
    # nostack cell - "iter4_1_cell"

    # 3stack nucleus - "iter5_1_3stack_nuc"
    # nostack nucleus - "iter5_1_nuc"


    for in_folder,out_folder in exps:
        print(f"segmenting experiment from in_folder: {in_folder} to out_folder: {out_folder}")
        #this produces a unique id based on the input and output folder locations on disk. It's not pretty, but it ensures each local folder
        #is both different from one another but should be the same if the segmentation fails partway through
        key = hash((str(Path(in_folder).absolute()).lower(),str(Path(out_folder).absolute()).lower())) 
        local_folder_name = f"local/{key}" #make each folder to the input

        segment_experiment(in_folder,out_folder,"keras/iter4_1_3stack_cell" if not nuc_only else None,"keras/iter5_1_3stack_nuc" if not cell_only else None,
                           out_subfolder="3stack_masks_out",local_folder=f"local/",auto_sublocal=False)
