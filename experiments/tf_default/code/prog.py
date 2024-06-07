import tensorflow as tf
from typing import Dict

# print("TensorFlow version:", tf.__version__)

# Define inputs
inputs: Dict[str, tf.Tensor] = {'v0_0': tf.constant([1.0, 2.0])}  # Example input

# Define the TensorFlow model using tf.Module
class Model(tf.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define TensorFlow variables or layers here if needed

    @tf.function(jit_compile=True)  # Enable XLA compilation for this method
    def __call__(self, v0_0):
        return v0_0
    
class Model1(tf.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Define TensorFlow variables or layers here if needed

    @tf.function(jit_compile=False)  # Enable XLA compilation for this method
    def __call__(self, v0_0):
        return v0_0

# Initialize the model and run it with inputs
model = Model()
print('==== Eager mode ====')
ret_eager = model(**inputs)


model = Model1()
print('==== XLA JIT mode ====')
# Call the same function again; it will run using XLA if possible
ret_xla_jit = model(**inputs)

print('==== Check ====')
# Check if the outputs from Eager mode and XLA JIT mode are close
tf.test.TestCase().assertAllClose(ret_eager, ret_xla_jit, atol=1e-3)

print('OK!')
