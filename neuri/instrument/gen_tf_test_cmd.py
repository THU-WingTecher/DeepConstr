import glob
import os
import sys

from natsort import natsorted

timeout = "15m"
tensorflow_src = sys.argv[1]  # "/path/to/tensorflow_src"

test_files = glob.glob(tensorflow_src + "/**/*_test.py", recursive=True)
test_files = natsorted(test_files)

print("#!/bin/bash")
print("set -x\n\n")

print(
    f"timeout {timeout} python {os.path.join(tensorflow_src, 'tensorflow/tools/docs/tf_doctest.py')}"
)

for test_file_path in test_files:
    print(f"timeout {timeout} python {test_file_path}")

print('\n\necho "Done"')
