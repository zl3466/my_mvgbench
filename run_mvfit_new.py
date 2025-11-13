"""
run 3dgs fitting from generated multi-view images
Improved version with parallel processing support for better GPU utilization
"""

import os, sys
from glob import glob
import os.path as osp
import multiprocessing as mp
import time

import numpy as np
from tqdm import tqdm

REDO = False

def process_single_sample(folder, args_dict, worker_id, port):
    """Process a single sample: train then render"""
    folder_name = osp.basename(folder)
    
    # Train
    train_cmd = f'python train.py --port {port} -s "{folder}" -lib lgm'
    if not args_dict.get('debug', False):
        train_cmd += ' --quiet '
    if args_dict.get('white_background', False):
        train_cmd += ' --white_background '
    
    code = os.system(train_cmd)
    if code != 0:
        print(f"Error training {folder_name} (worker {worker_id})")
        return False
    
    # Render
    render_cmd = f'python render.py -m "{folder}" --resolution 256 --elev_offset {args_dict.get("elev_offset", -10)}'
    if not args_dict.get('debug', False):
        render_cmd += ' --quiet '
    
    code = os.system(render_cmd)
    if code != 0:
        print(f"Error rendering {folder_name} (worker {worker_id})")
        return False
    
    return True

def process_sample_batch(args_tuple):
    """Process a batch of samples assigned to a worker"""
    sample_folders, args_dict, worker_id, base_port, progress_queue = args_tuple
    worker_port = base_port + worker_id
    
    for folder in sample_folders:
        success = process_single_sample(folder, args_dict, worker_id, worker_port)
        if progress_queue:
            folder_name = osp.basename(folder)
            progress_queue.put((folder_name, success))

def run_combined_parallel(args):
    """Process samples in parallel using multiple workers on the same GPU"""
    # Get all folders matching the pattern
    if osp.isdir(args.pat):
        folders = [args.pat]
    else:
        folders = sorted(glob(args.pat))
    
    if args.end is not None:
        folders = folders[:args.end]
    if args.start > 0:
        folders = folders[args.start:]
    
    if len(folders) == 0:
        print("No folders found matching pattern")
        return
    
    print(f"Found {len(folders)} folders to process")
    
    # Prepare args dict
    args_dict = {
        'debug': args.debug,
        'white_background': args.white_background,
        'elev_offset': args.elev_offset,
    }
    
    # Determine number of workers
    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = min(args.max_workers, len(folders), 8)
        if len(folders) < 4:
            num_workers = len(folders)
    
    if num_workers > 1:
        print(f"Using {num_workers} workers on GPU 0")
        
        # Split folders into batches
        batch_size = max(1, len(folders) // num_workers)
        batches = [folders[i:i + batch_size] for i in range(0, len(folders), batch_size)]
        
        # Print batch distribution
        print(f"\nBatch distribution:")
        for i, batch in enumerate(batches):
            batch_names = [osp.basename(f) for f in batch]
            print(f"  Worker {i}: {len(batch)} samples - {', '.join(batch_names[:3])}{'...' if len(batch_names) > 3 else ''}")
        print()
        
        # Setup parallel processing with Manager for Queue sharing
        base_port = 6000
        manager = mp.Manager()
        progress_queue = manager.Queue()
        worker_args = [(batch, args_dict, i, base_port, progress_queue) for i, batch in enumerate(batches)]
        
        # Process in parallel with progress tracking
        with mp.Pool(processes=num_workers) as pool:
            result = pool.map_async(process_sample_batch, worker_args)
            
            # Monitor progress
            completed = 0
            with tqdm(total=len(folders), desc="Processing samples") as pbar:
                while completed < len(folders):
                    try:
                        while not progress_queue.empty():
                            folder_name, success = progress_queue.get_nowait()
                            completed += 1
                            pbar.update(1)
                            pbar.set_postfix({'current': folder_name[:30]})
                    except:
                        pass
                    
                    if result.ready():
                        # Process remaining items
                        while not progress_queue.empty():
                            try:
                                progress_queue.get_nowait()
                                completed += 1
                                pbar.update(1)
                            except:
                                break
                        break
                    
                    time.sleep(0.1)
            
            result.get()
    else:
        # Sequential processing
        base_port = 6000
        for folder in tqdm(folders, desc="Processing samples"):
            process_single_sample(folder, args_dict, 0, base_port)
    
    # Alignment (same as original)
    if args.align:
        print("Aligning 3dgs")
        folder = osp.basename(osp.dirname(args.pat))
        cmd = f'python eval/align_3dgs.py -ft output/consistency/sv3dp+mvpnet50-sv3d-v21-elev030+i000_even -fs "output/consistency/{folder}even"'
        print(cmd)
        os.system(cmd)
        cmd = f'python render.py -m "{args.pat}" --quiet --resolution 256 --normalize_gs '
        os.system(cmd)

def run_combined(args):
    """all folder in one train and one render command, first 3dgs training, then render"""
    if not args.white_background:
        assert '+' not in args.pat, f'please check if black background is indeed the case for {args.pat}!'
    cmd = f'python train.py --port {np.random.randint(6000, 6500)} -s "{args.pat}" -lib lgm --start {args.start}'
    if args.end is not None:
        cmd += f' --end {args.end}'
    if not args.debug:
        cmd += ' --quiet '
    if args.white_background:
        cmd += ' --white_background '
    print(f"train cmd: {cmd}")
    code = os.system(cmd)
    if code != 0:
        exit(code)

    # Now render
    cmd = f'python render.py -m "{args.pat}" --resolution 256 --elev_offset {args.elev_offset} --start {args.start}' # training gt images are resized to 256 as well
    if args.end is not None:
        cmd += f' --end {args.end}'
    if not args.debug:
        cmd += ' --quiet '
    print(f"render cmd: {cmd}")
    code = os.system(cmd)
    if code != 0 :
        exit(code)

    if args.align:
        print("Aligning 3dgs")
        folder = osp.basename(osp.dirname(args.pat))
        cmd = f'python eval/align_3dgs.py -ft output/consistency/sv3dp+mvpnet50-sv3d-v21-elev030+i000_even -fs "output/consistency/{folder}even"'
        print(cmd)
        os.system(cmd)
        # re-render
        cmd = f'python render.py -m "{args.pat}" --quiet --resolution 256 --normalize_gs '
        os.system(cmd)


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pat') # eg. example/imagedream-v21-nolight_even/*
    parser.add_argument('--white_background', default=False, action='store_true') # for predictions
    parser.add_argument('-b', '--bbox_size', type=float, default=5.0)
    parser.add_argument('--align', default=False, action='store_true')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-eo', '--elev_offset', default=-10, type=float)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    # Parallel processing options
    parser.add_argument('--parallel', default=False, action='store_true', 
                        help='Enable parallel processing of multiple samples on the same GPU')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of parallel workers (0 = auto-detect, default: up to 8)')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of workers to use (default: 8)')

    args = parser.parse_args()

    if args.parallel:
        run_combined_parallel(args)
    else:
        run_combined(args)
