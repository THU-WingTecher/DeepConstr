# Data collection

## Instrumentation

### PyTorch

1. Place lines in `instr_torch.py` at the end of `<torch_path>/__init__.py`.
2. `git clone https://github.com/pytorch/pytorch.git pytorch_src && cd pytorch_src`
3. Run the following command, and collected traces will be saved to `<pt_traces>`.
    ```shell
    INSTR=1 OUT_DIR=<pt_traces> python test/run_test.py --continue-through-error --
    xdoctest-command all
    ```

### TensorFlow

1. Delete the following lines near the end of `<tf_path>/__init__.py`.
    ```python
    try:
      del python
    except NameError:
      pass
    try:
      del core
    except NameError:
      pass
    try:
      del compiler
    except NameError:
      pass
    ```
2. Place lines in `instr_tf.py` at the end of `<tf_path>/__init__.py`.
3. `git clone https://github.com/tensorflow/tensorflow.git /path/to/tf_src`
4. Run the following commands, and collected traces will be saved to `<tf_traces>`.
    ```shell
    INSTR=1 CUDA_VISIBLE_DEVICES='' OUT_DIR=<tf_traces> python /path/to/tf_src/tensorflow/tools/docs/tf_doctest.py
    python <autoinf_repo>/autoinf/instrument/gen_tf_test_cmd.py /path/to/tf_src > run_tf_tests.sh
    INSTR=1 CUDA_VISIBLE_DEVICES='' OUT_DIR=<tf_traces> bash run_tf_tests.sh
    ```

## Normalization

```shell
CUDA_VISIBLE_DEVICES='' python autoinf/instrument/collect.py <torch_traces> <torch_normed_traces> torch
CUDA_VISIBLE_DEVICES='' python autoinf/instrument/categorize.py <torch_normed_traces> fix_dim torch
```

Data for fuzzing will be saved to `<torch_normed_traces>/cat_fix_dim`.
