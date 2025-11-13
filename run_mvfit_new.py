"""
run 3dgs fitting from generated multi-view images
Improved version with parallel processing support for better GPU utilization
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
from tqdm import tqdm

REDO = False

def process_single_sample(folder, args_dict, worker_id, port):
    """Process a single sample: train then render"""
    folder_name = osp.basename(folder)
    
    # Train - use subprocess to ensure proper process isolation
    train_cmd = ['python', 'train.py', '--port', str(port), '-s', folder, '-lib', 'lgm']
    if not args_dict.get('debug', False):
        train_cmd.append('--quiet')
    if args_dict.get('white_background', False):
        train_cmd.append('--white_background')
    
    print(f"Worker {worker_id} running train command: {' '.join(train_cmd)}", flush=True)
    
    # Run training - don't capture output to see it in real-time, but redirect stderr
    try:
        result = subprocess.run(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True, timeout=3600)  # 1 hour timeout per training
        if result.returncode != 0:
            print(f"Error training {folder_name} (worker {worker_id}): {result.stderr[:500]}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout training {folder_name} (worker {worker_id})", flush=True)
        return False
    except Exception as e:
        print(f"Exception training {folder_name} (worker {worker_id}): {e}", flush=True)
        return False
    
    # Render
    render_cmd = ['python', 'render.py', '-m', folder, '--resolution', '256', 
                  '--elev_offset', str(args_dict.get('elev_offset', -10))]
    if not args_dict.get('debug', False):
        render_cmd.append('--quiet')
    
    print(f"Worker {worker_id} running render command: {' '.join(render_cmd)}", flush=True)
    
    try:
        result = subprocess.run(render_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True, timeout=600)  # 10 min timeout per render
        if result.returncode != 0:
            print(f"Error rendering {folder_name} (worker {worker_id}): {result.stderr[:500]}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout rendering {folder_name} (worker {worker_id})", flush=True)
        return False
    except Exception as e:
        print(f"Exception rendering {folder_name} (worker {worker_id}): {e}", flush=True)
        return False
    
    return True

def process_sample_batch(args_tuple):
    """Process a batch of samples assigned to a worker"""
    sample_folders, args_dict, worker_id, base_port, progress_queue = args_tuple
    worker_port = base_port + worker_id
    
    try:
        # Log worker start and signal that worker is alive
        print(f"Worker {worker_id} started processing {len(sample_folders)} samples (PID: {os.getpid()})", flush=True)
        if progress_queue:
            progress_queue.put(('_worker_started', worker_id))
        
        for folder in sample_folders:
            folder_name = osp.basename(folder)
            print(f"Worker {worker_id} starting {folder_name}...", flush=True)
            start_time = time.time()
            
            try:
                success = process_single_sample(folder, args_dict, worker_id, worker_port)
                elapsed = time.time() - start_time
                print(f"Worker {worker_id} completed {folder_name} in {elapsed:.1f}s", flush=True)
                
                if progress_queue:
                    progress_queue.put((folder_name, success))
            except Exception as e:
                print(f"Worker {worker_id} ERROR processing {folder_name}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                if progress_queue:
                    progress_queue.put((folder_name, False))
        
        print(f"Worker {worker_id} finished all samples", flush=True)
    except Exception as e:
        print(f"Worker {worker_id} FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

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
    
    # Determine number of workers based on available CPUs
    # Each worker needs at least 1 CPU core, and we want some headroom for the main process
    available_cpus = mp.cpu_count()
    
    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        # Limit workers to available CPUs (leave 1-2 CPUs for main process and system)
        max_workers_by_cpu = max(1, available_cpus - 2)
        num_workers = min(args.max_workers, len(folders), max_workers_by_cpu, 8)
        if len(folders) < 4:
            num_workers = len(folders)
    
    if num_workers > 1:
        print(f"Available CPUs: {available_cpus}, Using {num_workers} workers on GPU 0")
        print(f"NOTE: Multiple processes sharing one GPU may serialize if GPU memory is limited.")
        print(f"      Each training process needs GPU memory - monitor with 'nvidia-smi' to verify parallel execution.\n")
        
        # Split folders into batches
        batch_size = max(1, len(folders) // num_workers)
        batches = [folders[i:i + batch_size] for i in range(0, len(folders), batch_size)]
        
        # Print batch distribution
        print(f"Batch distribution:")
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
            
            # Monitor progress with timeout
            completed = 0
            no_progress_count = 0
            max_no_progress = 100  # 10 seconds without progress before checking if workers are stuck
            
            # Wait a moment for workers to start
            print("Waiting for workers to start...", flush=True)
            time.sleep(2)
            
            with tqdm(total=len(folders), desc="Processing samples") as pbar:
                workers_started = set()
                while completed < len(folders):
                    try:
                        # Try to get progress update with timeout
                        try:
                            folder_name, status = progress_queue.get(timeout=0.1)
                            
                            # Handle worker start signals
                            if folder_name == '_worker_started':
                                workers_started.add(status)
                                print(f"Worker {status} confirmed started. Total started: {len(workers_started)}/{num_workers}", flush=True)
                                continue
                            
                            # Normal progress update
                            completed += 1
                            pbar.update(1)
                            pbar.set_postfix({'current': folder_name[:30]})
                            no_progress_count = 0  # Reset counter on progress
                        except Empty:
                            # No progress in this iteration (timeout)
                            no_progress_count += 1
                            
                            # Check if workers are done
                            if result.ready():
                                # Process any remaining items
                                while True:
                                    try:
                                        folder_name, success = progress_queue.get_nowait()
                                        if folder_name != '_worker_started':
                                            completed += 1
                                            pbar.update(1)
                                    except (Empty, ValueError):
                                        break
                                break
                            
                            # If no progress for too long, check if workers are still alive
                            if no_progress_count >= max_no_progress:
                                print(f"\nWarning: No progress for {max_no_progress * 0.1:.1f}s.", flush=True)
                                print(f"Workers started: {len(workers_started)}/{num_workers}", flush=True)
                                print(f"Completed samples: {completed}/{len(folders)}", flush=True)
                                print("Check if training processes are running with: ps aux | grep train.py", flush=True)
                                no_progress_count = 0  # Reset to avoid spam
                    except KeyboardInterrupt:
                        print("\nInterrupted by user. Terminating workers...", flush=True)
                        pool.terminate()
                        pool.join()
                        raise
                    except Exception as e:
                        print(f"\nError in progress monitoring: {e}", flush=True)
                        # Continue monitoring
            
            # Wait for all workers to complete
            try:
                result.get(timeout=1)
            except:
                pass  # Workers should be done by now
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
                        help='Number of parallel workers (0 = auto-detect based on available CPUs, default: up to 8)')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of workers to use (default: 8, limited by available CPUs)')

    args = parser.parse_args()

    if args.parallel:
        run_combined_parallel(args)
    else:
        run_combined(args)
