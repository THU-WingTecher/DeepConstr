import json
import subprocess
import os
from typing import Literal
import hydra
import concurrent.futures
import subprocess
import threading
import multiprocessing
from experiments.evaluate_models import model_exec, batched
from experiments.evaluate_tf_models import tf_model_exec, clear_gcda
from neuri.logger import DTEST_LOG
# Load the JSON file

BASELINES = ["symbolic-cinit", "neuri", "constrinf", "constrinf_2"]
FIXED_FUNC = None #"torch.sin"#"tf.cos"#"torch.sin"
cov_parallel = 1

def activate_conda_environment(env_name):
    """
    Activates the specified conda environment.

    :param env_name: Name of the conda environment to activate.
    """
    activation_command = "source /opt/conda/etc/profile.d/conda.sh && conda activate " + env_name
    subprocess.run(activation_command, shell=True, executable='/bin/bash')

def torch_batch_exec(batch, time2path, cov_save, model_type, backend_type, backend_target, *args, **kwargs):
    batch_paths = [time2path[time] for time in batch]
    profraw_path = os.path.join(cov_save, f"{max(batch)}.profraw")
    model_exec(
        batch_paths,
        model_type,
        backend_type,
        backend_target,
        profraw_path,
    )

def tf_batch_exec(batch, time2path, cov_save, model_type, backend_type, backend_target, root_path):
    batch_paths = [time2path[time] for time in batch]
    profraw_path = os.path.join(cov_save, f"{max(batch)}.info")
    tf_model_exec(
        batch_paths,
        model_type,
        backend_type,
        backend_target,
        profraw_path,
        root_path.replace('/','.')
    )

def parallel_collect_cov(root, model_type, backend_type, batch_size=100, backend_target="cpu", parallel=8):
    """
    Collects coverage data after fuzzing.
    
    :param root: Folder to all the tests.
    :param model_type: Model type used in fuzzing.
    :param backend_type: Backend type used in fuzzing.
    :param batch_size: Size of each batch for processing.
    :param backend_target: Backend target (cpu or cuda).
    :param parallel: Number of processes for execution.
    """

    time2path = {}
    for dir in os.listdir(root):
        if dir != "coverage":
            time2path[float(dir)] = os.path.join(root, dir)

    time_stamps = sorted(time2path.keys())
    # batch_size = len(time2path)
    # batches = [time_stamps[i:i + batch_size] for i in range(0, len(time_stamps), batch_size)]
    batches = [time_stamps] # no need to batch
    # print(f"=> Number of batches: {len(batches)} of size {batch_size}")
    print(f"=> Number of batches: {len(batches)} of size {len(time_stamps)}")

    cov_save = os.path.join(root, "coverage")
    if not os.path.exists(cov_save):
        os.mkdir(cov_save)
    if model_type == "tensorflow" :
        clear_gcda()
        batch_exec = tf_batch_exec
    elif model_type == "torch" :
        batch_exec = torch_batch_exec
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(BASELINES)) as executor:
        # Pass necessary arguments to the api_worker function
        # print('print', batch_exec, cov_save, model_type, backend_type, backend_target)
        futures = [executor.submit(batch_exec, batch, time2path, cov_save, model_type, backend_type, backend_target, root_path=root) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occured: {e}")

def process_profraw(path):

    activation_command = "source /opt/conda/etc/profile.d/conda.sh && conda activate " + "cov" + " && "
    arguments = [
        "python",
        "experiments/process_profraws.py",
        f"--root {path}",
        "--llvm-config-path $(which llvm-config-14)",
        '--instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so"',
        f"--batch-size 0 --parallel {cov_parallel}",
    ]
    full_command = activation_command + " ".join(arguments)

    p = subprocess.Popen(
        full_command,  # Show all output
        shell=True,
        executable='/bin/bash',
    )

    p.communicate()
    exit_code = p.returncode

    if exit_code != 0:
        print(
            f"==> process_profraw crashed when generating {os.path.join(path, 'coverage','merged_cov.pkl')}! => EXIT CODE {exit_code}"
        )    
def process_lcov(path):
    activation_command = "source /opt/conda/etc/profile.d/conda.sh && conda activate " + "cov" + " && "
    arguments = [
        "python",
        "experiments/process_lcov.py",
        f"--root {path}",
        f"--batch-size 0 --parallel {cov_parallel}",
    ]
    full_command = activation_command + " ".join(arguments)
    print(full_command)
    p = subprocess.Popen(
        full_command,  # Show all output
        shell=True,
        executable='/bin/bash',
    )

    p.communicate()
    exit_code = p.returncode

    if exit_code != 0:
        print(
            f"==> process_lcov crashed when generating {os.path.join(path, 'coverage','merged_cov.pkl')}! => EXIT CODE {exit_code}"
        )
def parallel_eval(api_list, BASELINES, config, task):
    """
    Runs fuzzing processes in parallel for each combination of API and baseline.
    
    :param api_list: List of APIs to fuzz.
    :param baseline_list: List of baseline methods to use.
    :param config: Configuration parameters for the fuzzing process.
    """

    with concurrent.futures.ProcessPoolExecutor(max_workers=config["exp"]["parallel"]) as executor:
        # Pass necessary arguments to the api_worker function
        futures = [executor.submit(api_worker, api, config, BASELINES, task) for api in api_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occured: {e}")

def api_worker(api, cfg, BASELINES, task):
    """
    Top-level worker function for multiprocessing, handles each API task.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(BASELINES)) as executor:
        # Pass necessary arguments to the api_worker function
        futures = [executor.submit(run, api, baseline, cfg, task) for baseline in BASELINES]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occured: {e}")
    # Trigger drawing after completing all baseline tasks for the current API
    # run_draw_script(api, cfg, BASELINES)

def run(api_name, baseline, config, task : Literal["fuzz", "cov"] = "cov"):
    """
    Runs the fuzzing process for a given API and baseline with the specified configuration.
    Captures and displays output in real-time.
    
    :param api_name: The name of the API to fuzz.
    :param baseline: The baseline method to use.
    :param config: Configuration parameters for the fuzzing process.
    :param max_retries: Maximum number of retries in case of failure.
    """
    tries = 0
    print(f"Running {task} API {api_name} with baseline {baseline}")
    test_pool = [FIXED_FUNC, api_name]
    test_pool_modified = '-'.join(test_pool)
    if config['mgen']['max_nodes'] is not None :
        max_nodes = config['mgen']['max_nodes']
    else :
        max_nodes = 3
    save_path = f"{os.getcwd()}/{config['exp']['save_dir']}/{config['model']['type']}-{baseline}-n{max_nodes}-{test_pool_modified}.models"
    def execute_command(command):
        """
        Executes a given command and prints its output in real-time.
        """
        print("Running\n", command)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        p.communicate()
        exit_code = p.returncode
        if exit_code != 0:
            return 
    
    if config['model']['type'] == "tensorflow":
        if baseline == "constrinf":
            RECORD = config["mgen"]["record_path"]
        elif baseline == "constrinf_2":
            RECORD = config["mgen"]["record_path"].replace("records", "only_acc")
        else:
            RECORD = os.path.join(os.getcwd(), "data", "tf_records")
    elif config['model']['type'] == "torch":
        if baseline == "constrinf":
            RECORD = config["mgen"]["record_path"]
        elif baseline == "constrinf_2":
            RECORD = config["mgen"]["record_path"].replace("records", "only_acc")
        else:
            RECORD = os.path.join(os.getcwd(), "data", "torch_records")
    # Construct the command to run fuzz.py
    fuzz_command = f"PYTHONPATH=$(pwd):$(pwd)/neuri python neuri/cli/fuzz.py " \
                   f"fuzz.time={config['fuzz']['time']} " \
                   f"mgen.record_path={RECORD} " \
                   f"fuzz.root=$(pwd)/{config['exp']['save_dir']}/{config['model']['type']}-{baseline}-n{max_nodes}-{test_pool_modified} " \
                   f"fuzz.save_test={save_path} " \
                   f"model.type={config['model']['type']} backend.type={config['backend']['type']} filter.type=\"[nan,dup,inf]\" " \
                    f"debug.viz=true hydra.verbose=fuzz fuzz.resume=true " \
                   f"mgen.method={baseline.split('_')[0]} mgen.max_nodes={max_nodes} mgen.test_pool=\"{test_pool}\""

    if task == "fuzz":
        # while tries < 30:
        execute_command(fuzz_command)
            # tries += 1
    elif task == "cov":
        print(f"Collect Cov for {api_name} with baseline {baseline}")
        print("Activate Conda env -cov")
        activate_conda_environment("cov")
        parallel_collect_cov(root=save_path,
                    model_type=config['model']['type'],
                    backend_type=config['backend']['type'],
                    batch_size=100,  # or other desired default
                    backend_target="cpu",  # or config-specified
                    parallel=cov_parallel)  # or other desired default
        if config['model']['type'] == "torch":
            process_profraw(save_path)
        elif config['model']['type'] == "tensorflow":
            process_lcov(save_path)
        else :
            raise NotImplementedError
    else :
        raise NotImplementedError
def run_draw_script(api_name : str, cfg, BASELINES):

    print(f"drawing with {api_name}")
    folder_names = [f"$(pwd)/{cfg['exp']['save_dir']}/{cfg['model']['type']}-{method}-n{cfg['mgen']['max_nodes']}-{'-'.join([FIXED_FUNC, api_name])}.models/coverage " for method in BASELINES]
    command = (
        f"""python experiments/viz_merged_cov.py """
        f"""--folders {' '.join([f"{fnm}" for fnm in folder_names])} """
        f"""--tags {' '.join([f"{nm}" for nm in BASELINES])} """
        f"--name {api_name}"
    )
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def load_api_names_from_data(record_path, pass_rate) : 
    from neuri.constrinf import make_record_finder  
    records = make_record_finder(
        path=record_path,
        pass_rate=pass_rate,
    )
    return [
        record.get("name") for record in records
    ]

def load_api_names_from_json(path) :
    with open(path, 'r') as file:
        api_sets = json.load(file)
    return api_sets
# Run the script for each API set

@hydra.main(version_base=None, config_path="../neuri/config", config_name="main")
def main(cfg) : 
    from neuri.cli.train import get_completed_list
    from experiments.summarize_merged_cov import exclude_intestable

    """
    totally, cfg['exp']['parallel'] * cov_parallel * len(BASELINES) process will be craeted
    """
    global FIXED_FUNC
    if cfg["model"]["type"] == "torch" :
        FIXED_FUNC = "torch.sin"
    elif cfg["model"]["type"] == "tensorflow" :
        FIXED_FUNC = "tf.cos"
    else :
        raise NotImplementedError
    # print(f" Will run {cfg['exp']['parallel'] * cov_parallel * len(BASELINES)} process in parallel")
    # api_names = load_api_names_from_data(cfg["mgen"]["record_path"], cfg["mgen"]["pass_rate"])
    # api_names = set(api_names)
    # print(f"From {len(api_names)} apis in total", sep=" ")

    # api_names = api_names[:100]     
    # api_names = list(set(['tf.keras.layers.Relu', 'tf.keras.layers.LeakyReLU', 'tf.sigmoid', 'tf.cos', 'tf.tan', 'tf.where', 'tf.multiply', 'tf.divide', 'tf.maximum', 'tf.minimum', 'tf.equal', 'tf.less', 'tf.logical_or', 'tf.math.logical_xor', 'tf.pow', 'tf.math.ceil', 'tf.round', 'tf.sqrt', 'tf.experimental.numpy.log2', 'tf.negative', 'tf.keras.layers.BatchNormalization', 'tf.reshape', 'tf.transpose', 'tf.keras.layers.Dense', 'tf.nn.conv2d', 'tf.nn.atrous_conv2d', 'tf.nn.depthwise_conv2d', 'tf.nn.separable_conv2d', 'tf.nn.conv2d_transpose', 'tf.squeeze', 'tf.expand_dims', 'tf.math.argmin', 'tf.experimental.numpy.tril', 'tf.concat', 'tf.cast', 'tf.reverse', 'tf.linalg.cholesky', 'tf.linalg.eigh', 'tf.matmul', 'tf.raw_ops.Xlog1py', 'tf.raw_ops.AddN', 'tf.raw_ops.Identity', 'tf.raw_ops.LessEqual', 'tf.raw_ops.Log', 'tf.raw_ops.MaxPoolV2', 'tf.raw_ops.NonMaxSuppressionV3', 'tf.raw_ops.Conv2D', 'tf.raw_ops.SelfAdjointEigV2', 'tf.raw_ops.InTopKV2', 'tf.raw_ops.PadV2', 'tf.raw_ops.LowerBound', 'tf.raw_ops.ResizeBilinear', 'tf.raw_ops.ArgMin', 'tf.raw_ops.Mean', 'tf.raw_ops.Max', 'tf.raw_ops.Imag', 'tf.raw_ops.Pack', 'tf.raw_ops.MirrorPad', 'tf.raw_ops.StatelessRandomUniformFullIntV2', 'tf.raw_ops.AvgPool', 'tf.raw_ops.Angle', 'tf.raw_ops.Shape', 'tf.raw_ops.MaxPool', 'tf.raw_ops.Conv3DBackpropInputV2', 'tf.raw_ops.SelectV2', 'tf.raw_ops.QuantizeAndDequantizeV4', 'tf.raw_ops.TopKV2', 'tf.raw_ops.DepthwiseConv2dNative', 'tf.raw_ops.ConcatV2', 'tf.raw_ops.ResizeNearestNeighbor', 'tf.raw_ops.Conv2DBackpropFilter', 'tf.raw_ops.Where', 'tf.raw_ops.BroadcastArgs', 'tf.raw_ops.Complex', 'tf.raw_ops.RandomGammaGrad', 'tf.raw_ops.MatrixDiagV3', 'tf.raw_ops.FusedBatchNorm', 'tf.raw_ops.Any', 'tf.raw_ops.ExpandDims', 'tf.raw_ops.Unique', 'tf.raw_ops.AvgPool3D', 'tf.raw_ops.Lgamma', 'tf.raw_ops.ListDiff', 'tf.raw_ops.GatherV2', 'tf.raw_ops.Conv2DBackpropInput', 'tf.raw_ops.Cumsum', 'tf.raw_ops.ExtractImagePatches', 'tf.raw_ops.Bincount', 'tf.raw_ops.Xlogy', 'tf.raw_ops.IgammaGradA', 'tf.raw_ops.All', 'tf.raw_ops.Log1p', 'tf.raw_ops.ShapeN', 'tf.raw_ops.ReluGrad', 'tf.raw_ops.Roll', 'tf.raw_ops.MatrixDiagPartV3', 'tf.raw_ops.Igamma', 'tf.raw_ops.PlaceholderWithDefault', 'tf.raw_ops.StatelessMultinomial', 'tf.raw_ops.DepthwiseConv2dNativeBackpropInput', 'tf.raw_ops.DepthwiseConv2dNativeBackpropFilter', 'tf.raw_ops.MaxPool3D', 'tf.raw_ops.MaxPoolGrad', 'tf.raw_ops.Zeta', 'tf.raw_ops.MaxPool3DGrad', 'tf.raw_ops.Real', 'tf.raw_ops.MaxPool3DGradGrad', 'tf.raw_ops.TensorStridedSliceUpdate', 'tf.raw_ops.IdentityN', 'tf.raw_ops.Conv3D', 'tf.raw_ops.BatchMatMulV3', 'tf.raw_ops.RealDiv', 'tf.raw_ops.SparseToDense', 'tf.raw_ops.SpaceToDepth', 'tf.raw_ops.MatrixSetDiagV3', 'tf.raw_ops.ConjugateTranspose', 'tf.raw_ops.Pad', 'tf.raw_ops.ZerosLike', 'tf.raw_ops.MaxPoolGradGrad', 'tf.raw_ops.StatelessRandomUniformFullInt', 'tf.raw_ops.Tile', 'tf.raw_ops.TensorScatterUpdate', 'tf.raw_ops.Einsum']))
    api_names = list(set(['torch.nn.ReLU', 'torch.nn.GELU', 'torch.nn.LeakyReLU', 'torch.nn.PReLU', 'torch.logical_and', 'torch.logical_or', 'torch.logical_xor', 'torch.nn.Softmax', 'torch.nn.MaxPool2d', 'torch.nn.AvgPool2d', 'torch.nn.ConstantPad2d', 'torch.nn.ReflectionPad2d', 'torch.nn.ReplicationPad2d', 'torch.nn.BatchNorm2d', 'torch.nn.Conv1d', 'torch.nn.Conv2d', 'torch.reshape', 'torch.nn.functional.interpolate', 'torch.triu', 'torch.nn.Linear', 'torch.cat']+['torch.nn.functional.hinge_embedding_loss', 'torch.searchsorted', 'torch.conv2d', 'torch.triangular_solve', 'torch.mv', 'torch._C._nn.fractional_max_pool3d', 'torch.Tensor.random_', 'torch.column_stack', 'torch._C._special.special_erfcx', 'torch.angle', 'torch.nn.functional.fractional_max_pool2d_with_indices', 'torch.ormqr', 'torch.Tensor.copy_', 'torch.symeig', 'torch._C._linalg.linalg_cross', 'torch.polygamma', 'torch._C._special.special_hermite_polynomial_h', 'torch._C._nn.max_pool3d_with_indices', 'torch.bilinear', 'torch._C._special.special_modified_bessel_k0', 'torch.Tensor.select', 'torch.rand_like', 'torch._C._special.special_polygamma', 'torch.unbind', 'torch._C._special.special_modified_bessel_i1', 'torch._C._linalg.linalg_householder_product', 'torch.nn.functional.grid_sample', 'torch.Tensor.uniform_', 'torch.nn.functional.fractional_max_pool3d_with_indices', 'torch.nn.functional.cross_entropy', 'torch.Tensor.unbind', 'torch.index_put', 'torch.pixel_shuffle', 'torch.Tensor.new_empty_strided', 'torch.slice_scatter', 'torch.einsum', 'torch.randn_like', 'torch._C._nn.linear', 'torch.randint_like', 'torch.hstack', 'torch.permute', 'torch.avg_pool1d', 'torch._C._linalg.linalg_multi_dot', 'torch.lu_solve', 'torch.logcumsumexp', 'torch.conv_transpose3d', 'torch.polar', 'torch._C._linalg.linalg_lu_solve', 'torch.as_strided_scatter', 'torch.all', 'torch.Tensor.sum_to_size', 'torch.conv_transpose2d', 'torch.baddbmm', 'torch.istft', 'torch.Tensor.normal_', 'torch.pixel_unshuffle', 'torch.max_pool1d', 'torch.amax', 'torch.nn.functional.embedding', 'torch.broadcast_tensors', 'torch.nn.functional.feature_alpha_dropout', 'torch.Tensor.stft', 'torch.outer', 'torch.Tensor.contiguous', 'torch.addr', 'torch.split', 'torch.Tensor.split', 'torch.nn.functional.dropout1d', 'torch.triplet_margin_loss', 'torch.meshgrid', 'torch.nn.functional.unfold', 'torch.nn.functional.max_pool1d_with_indices', 'torch.logdet', 'torch.dropout', 'torch.conv1d', 'torch.nn.functional.dropout2d', 'torch.addmm', 'torch.nn.functional.multi_head_attention_forward', 'torch.addmv', 'torch.native_dropout', 'torch.nn.functional.pad', 'torch.nn.functional.glu', 'torch.stack', 'torch._C._nn.flatten_dense_tensors', 'torch.native_batch_norm', 'torch._C._nn.unflatten_dense_tensors', 'torch.max_pool1d_with_indices', 'torch.true_divide', 'torch.aminmax', 'torch.cholesky_solve', 'torch.cartesian_prod', 'torch.Tensor.requires_grad_', 'torch.gather', 'torch.Tensor.index_fill', 'torch.max_pool3d', 'torch.Tensor.logical_and_', 'torch.logical_not', 'torch.max_pool2d', 'torch.addbmm', 'torch.nn.functional.dropout', 'torch.conv_transpose1d', 'torch.diagonal_scatter', 'torch._C._special.special_entr', 'torch.nn.functional.max_pool3d_with_indices', 'torch._C._nn.fractional_max_pool2d', 'torch.pdist', 'torch._C._nn.pad_sequence', 'torch.empty_like', 'torch.vstack', 'torch.nn.functional.dropout3d', 'torch._C._special.special_modified_bessel_i0', 'torch.block_diag', 'torch.logit', 'torch.Tensor.new_empty', 'torch.amin', 'torch.dstack', 'torch.pinverse']))
    completed = exclude_intestable()
    print(completed)
    for name in completed :
        if name in api_names :
            api_names.remove(name)
    print(api_names)          
    print(f"Test {len(api_names)} apis in total", sep=" ")
    # api_names = load_api_names_from_json("/artifact/tests/test.json")
    parallel_eval(api_names, BASELINES, cfg, task="fuzz")
    # parallel_eval(api_names, BASELINES, cfg, task="cov")
if __name__ == "__main__":
    main()
