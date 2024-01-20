import tensorflow as tf

# Path where the model is saved
model_path = '/artifact/gen/tensorflow-neuri-n5.models/315.530/model/tfnet'

# Load the model
loaded_model = tf.saved_model.load(model_path)

# Get the inference function (signature)
inference_func = loaded_model.signatures["serving_default"]

# Dummy input data (replace with your actual data)
# Make sure the shape and type match the input your model expects
input_data = tf.constant([[1.0, 2.0], [5.0, 6.0]])

# Run inference
output_data = inference_func(input_data)

# Output will be a dictionary, you can retrieve the output tensor by the key
result = output_data["output_key"]

# Perform additional operations with the result if needed
# ...

##pickle file -> GraphIR
# --IR
#     --inst 
#     --iexpr 
#     --op