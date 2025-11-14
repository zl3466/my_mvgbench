"""
run 3dgs fitting from generated multi-view images
Multi-GPU version for parallel training across multiple GPUs
"""

import os, sys
from glob import glob
import os.path as osp
import multiprocessing as mp
try:
    from queue import Empty
except ImportError:
    from Queue import Empty  # Python 2 compatibility
import time
import subprocess

import numpy as np
import torch
from tqdm import tqdm

REDO = False

def get_available_gpus():
    """Get the number of available GPUs"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU (not recommended).")
        return 0
    return torch.cuda.device_count()

def process_single_sample(folder, args_dict, gpu_id, port):
    """Process a single sample: train then render on a specific GPU"""
    folder_name = osp.basename(folder)
    
    # Set CUDA_VISIBLE_DEVICES to use only the specified GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Train - use subprocess to ensure proper process isolation
    train_cmd = ['python', 'train.py', '--port', str(port), '-s', folder, '-lib', 'lgm']
    if not args_dict.get('debug', False):
        train_cmd.append('--quiet')
    if args_dict.get('white_background', False):
        train_cmd.append('--white_background')
    
    # Run training with GPU assignment
    try:
        if args_dict.get('debug', False):
            result = subprocess.run(train_cmd, env=env, timeout=3600)
        else:
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run(train_cmd, env=env, stdout=devnull, stderr=subprocess.PIPE, 
                                       text=True, timeout=3600)
                if result.returncode != 0 and result.stderr:
                    print(f"Error training {folder_name} (GPU {gpu_id}): {result.stderr[:500]}", flush=True)
        
        if result.returncode != 0:
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout training {folder_name} (GPU {gpu_id})", flush=True)
        return False
    except Exception as e:
        print(f"Exception training {folder_name} (GPU {gpu_id}): {e}", flush=True)
        return False
    
    # Render - also use the same GPU
    render_cmd = ['python', 'render.py', '-m', folder, '--resolution', '256', 
                  '--elev_offset', str(args_dict.get('elev_offset', -10))]
    if not args_dict.get('debug', False):
        render_cmd.append('--quiet')
    
    try:
        if args_dict.get('debug', False):
            result = subprocess.run(render_cmd, env=env, timeout=600)
        else:
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run(render_cmd, env=env, stdout=devnull, stderr=subprocess.PIPE, 
                                       text=True, timeout=600)
                if result.returncode != 0 and result.stderr:
                    print(f"Error rendering {folder_name} (GPU {gpu_id}): {result.stderr[:500]}", flush=True)
        
        if result.returncode != 0:
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout rendering {folder_name} (GPU {gpu_id})", flush=True)
        return False
    except Exception as e:
        print(f"Exception rendering {folder_name} (GPU {gpu_id}): {e}", flush=True)
        return False
    
    return True

def worker_process(args_tuple):
    """Worker process that pulls folders from a queue and processes them on a specific GPU"""
    work_queue, args_dict, gpu_id, base_port, progress_queue = args_tuple
    
    # Set CUDA_VISIBLE_DEVICES for this worker process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Each worker gets a range of ports to avoid conflicts
    # Ports: base_port + gpu_id * 100 + [0-99] for this GPU
    port_base = base_port + gpu_id * 100
    port_counter = 0
    
    while True:
        try:
            # Get next folder from queue (with timeout to check for sentinel)
            try:
                folder = work_queue.get(timeout=1)
            except Empty:
                continue
            
            # Check for sentinel value indicating no more work
            if folder is None:
                break
            
            folder_name = osp.basename(folder)
            # Assign unique port for this subprocess
            worker_port = port_base + (port_counter % 100)
            port_counter += 1
            
            try:
                success = process_single_sample(folder, args_dict, gpu_id, worker_port)
                if progress_queue:
                    progress_queue.put((folder_name, success, gpu_id))
            except Exception as e:
                print(f"GPU {gpu_id} ERROR processing {folder_name}: {e}", flush=True)
                if progress_queue:
                    progress_queue.put((folder_name, False, gpu_id))
            finally:
                work_queue.task_done()
        except Exception as e:
            print(f"GPU {gpu_id} FATAL ERROR: {e}", flush=True)

def run_combined_multigpu(args):
    """Process samples in parallel using multiple GPUs"""
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
    
    # Get available GPUs
    num_gpus = get_available_gpus()
    if num_gpus == 0:
        print("No GPUs available. Please check your CUDA setup.")
        return
    
    # Determine number of GPUs to use
    if args.num_gpus > 0:
        num_gpus_to_use = min(args.num_gpus, num_gpus)
    else:
        num_gpus_to_use = num_gpus
    
    if num_gpus_to_use > len(folders):
        num_gpus_to_use = len(folders)
        print(f"More GPUs than folders. Using {num_gpus_to_use} GPUs.")
    
    print(f"Using {num_gpus_to_use} GPU(s) out of {num_gpus} available\n")
    
    # Prepare args dict
    args_dict = {
        'debug': args.debug,
        'white_background': args.white_background,
        'elev_offset': args.elev_offset,
    }
    
    if num_gpus_to_use > 1:
        # Use a work queue for dynamic load balancing
        # Workers pull folders as they become available
        base_port = 6000
        manager = mp.Manager()
        work_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        # Put all folders into the work queue
        for folder in folders:
            work_queue.put(folder)
        
        # Add sentinel values to signal workers to stop
        for _ in range(num_gpus_to_use):
            work_queue.put(None)
        
        # Create worker processes
        worker_args = [(work_queue, args_dict, i, base_port, progress_queue) for i in range(num_gpus_to_use)]
        processes = []
        
        for args_tuple in worker_args:
            p = mp.Process(target=worker_process, args=(args_tuple,))
            p.start()
            processes.append(p)
        
        # Monitor progress
        completed = 0
        try:
            with tqdm(total=len(folders), desc="Processing samples") as pbar:
                while completed < len(folders):
                    try:
                        folder_name, success, gpu_id = progress_queue.get(timeout=0.5)
                        completed += 1
                        status = "✓" if success else "✗"
                        pbar.update(1)
                        pbar.set_postfix({'current': folder_name[:30], 'GPU': gpu_id, 'status': status})
                    except Empty:
                        # Check if any processes are still alive
                        if not any(p.is_alive() for p in processes):
                            # Process any remaining items
                            while True:
                                try:
                                    folder_name, success, gpu_id = progress_queue.get_nowait()
                                    completed += 1
                                    pbar.update(1)
                                except (Empty, ValueError):
                                    break
                            break
        except KeyboardInterrupt:
            print("\nInterrupted by user. Terminating workers...", flush=True)
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
            raise
        
        # Wait for all workers to complete
        for p in processes:
            p.join()
    else:
        # Single GPU processing
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
    # Multi-GPU options
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='Number of GPUs to use (0 = use all available GPUs, default: 0)')
    parser.add_argument('--single_gpu', default=False, action='store_true',
                        help='Disable multi-GPU and run sequentially on a single GPU')

    args = parser.parse_args()

    if args.single_gpu:
        run_combined(args)
    else:
        run_combined_multigpu(args)

