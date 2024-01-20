# import pickle
# import sys 
# sys.path.append('/artifact/neuri')
# err_path = '/artifact/gen/tensorflow-neuri-n1/bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-0/model/gir.pkl'

# with open(err_path, 'rb') as file:
#     # Load the object from the file
#     loaded_object = pickle.load(file)

# print(loaded_object)

# op
# loaded_object.insts[1].iexpr.op.inst.invoke_str(loaded_object.insts[1].iexpr.op.attrs)
# inputs
# loaded_object.insts[0].iexpr.args
# loaded_object.insts[0].iexpr.op
# loaded_object.insts[0].retvals()

import tensorflow as tf 

ip = tf.constant([1,4,6,1], dtype=tf.float32)


def forward(x) :
    x = tf.raw_ops.Cast(x=x, DstT=tf.int64, Truncate=False, name=None)
    return x 

opt = tf.function(jit_compile=True)(forward)

noop = forward(ip)
print(noop)
op = opt(ip)
tf.debugging.assert_near(noop, op, atol=0.001, rtol=0.001)
