# Rule Synthesizer

Make sure you have run `. ./install.sh`, which installes necessary third-party libraries and configures envrionment variables, at the root directory of the artifact.

Use

```bash
python3 tree.py [--dump_dir PATH]
```

to generate expression trees. If `dump_dir` is not specified, the file `tree.pkl` will be saved in `gen/` by default. The tree generation step takes about 7 minutes.

Use

```bash
python3 augmentation.py [--dump_dir PATH] [--parallel PNUM] [--library LIB]
```

to generate augmented records. By default, `PATH=gen/`, `PNUM=16`, `LIB=["tf", "torch"]`. The augmentation step takes less than 1 hour.
