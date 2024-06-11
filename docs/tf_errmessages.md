## Good Error Messages

 0. [tf.acos]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Acos}}; Op<name=Acos; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Acos] name: DmqO"
 1. [tf.add]{{function_node __wrapped__AddV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3] vs. [7,7,7,7] [Op:AddV2]"
 2. [tf.asin]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Asin}}; Op<name=Asin; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Asin] name: OCes"
 3. [tf.atan]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Atan}}; Op<name=Atan; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Atan] name: HXqt"
 4. [tf.clip_by_value]{{function_node __wrapped__Maximum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [2,9,9,9,9,9,9] vs. [3,7,6,1,9] [Op:Maximum]"
 5. [tf.clip_by_value]Shapes () and (7, 1, 6) are incompatible"
 6. [tf.clip_by_value]cannot compute Minimum as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Minimum]"
 7. [tf.cos]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Cos}}; Op<name=Cos; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Cos] name: ajEx"
 8. [tf.equal]{{function_node __wrapped__Equal_device_/job:localhost/replica:0/task:0/device:CPU:0}} Broadcast between [6,6,6,1,7,7] and [6,6,6,1,7,7,1] is not supported yet. [Op:Equal] name: aHQI"
 9. [tf.equal]negative dimensions are not allowed"
 10. [tf.equal]cannot compute Equal as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Equal] name: EvgX"
 11. [tf.equal]{{function_node __wrapped__Equal_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [7,2,5,7,2,6,7] vs. [1,7,3,7,2,7,7] [Op:Equal] name: TWxE"
 12. [tf.expand_dims]{{function_node __wrapped__ExpandDims_device_/job:localhost/replica:0/task:0/device:CPU:0}} Tried to expand dim index 52 for tensor with 7 dimensions. [Op:ExpandDims] name: OKOa"
 13. [tf.experimental.numpy.tril]Argument to tril must have rank at least 2"
 14. [tf.experimental.numpy.triu]Argument to triu must have rank at least 2"
 15. [tf.floor]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Floor}}; Op<name=Floor; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Floor] name: LtHI"
 16. [tf.gather]{{function_node __wrapped__GatherV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} indices = -4 is not in [0, 1) [Op:GatherV2] name: pkSL"
 17. [tf.gather]{{function_node __wrapped__GatherV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected batch_dims in the range [0, 0], but got 7 [Op:GatherV2] name: aFLm"
 18. [tf.gather]{{function_node __wrapped__GatherV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Shape must be at least rank 10 but is rank 7 [Op:GatherV2] name: KULl"
 19. [tf.greater]cannot compute Greater as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Greater] name: cpkO"
 20. [tf.greater]{{function_node __wrapped__Greater_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3,3,3,3,3,3] vs. [7,1,7,8,1,3,9] [Op:Greater] name: KrAG"
 21. [tf.less]cannot compute Less as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Less] name: yvPI"
 22. [tf.less]{{function_node __wrapped__Less_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [7,3,6,6,6] vs. [7,3,6,6] [Op:Less] name: xKFa"
 23. [tf.linalg.cholesky]{{function_node __wrapped__Cholesky_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input matrix must be square. [Op:Cholesky] name: CcuD"
 24. [tf.linalg.cholesky]{{function_node __wrapped__Cholesky_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input tensor 0 must have rank >= 2, got 0 [Op:Cholesky] name: TvDq"
 25. [tf.linalg.eigh]{{function_node __wrapped__SelfAdjointEigV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input matrix must be square. [Op:SelfAdjointEigV2] name: RDbM"
 26. [tf.linalg.eigh]{{function_node __wrapped__SelfAdjointEigV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input tensor 0 must have rank >= 2, got 0 [Op:SelfAdjointEigV2] name: Mbwg"
 27. [tf.logical_or]negative dimensions are not allowed"
 28. [tf.logical_or]{{function_node __wrapped__LogicalOr_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [7,2,2,5,4,5,5] vs. [2,7,7,7,7,7,7] [Op:LogicalOr] name: TQAA"
 29. [tf.logical_or]cannot compute LogicalOr as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:LogicalOr] name: PFdW"
 30. [tf.math.ceil]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Ceil}}; Op<name=Ceil; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Ceil] name: TBOU"
 31. [tf.math.logical_xor]{{function_node __wrapped__LogicalOr_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [5,1,5,0,2,7,0] vs. [1,7,7,1,4,1,1] [Op:LogicalOr]"
 32. [tf.math.logical_xor]cannot compute LogicalOr as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:LogicalOr]"
 33. [tf.math.reduce_max]{{function_node __wrapped__Max_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (-1 for input with 0 dimension(s) [Op:Max] name: KqHF"
 34. [tf.math.reduce_mean]{{function_node __wrapped__Mean_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (16 for input with 2 dimension(s) [Op:Mean] name: LzjI"
 35. [tf.math.reduce_min]{{function_node __wrapped__Min_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (8 for input with 7 dimension(s) [Op:Min] name: QjTw"
 36. [tf.math.reduce_prod]{{function_node __wrapped__Prod_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (67 for input with 4 dimension(s) [Op:Prod] name: GukZ"
 37. [tf.math.reduce_sum]{{function_node __wrapped__Sum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (51 for input with 2 dimension(s) [Op:Sum] name: JamY"
 38. [tf.math.subtract]{{function_node __wrapped__Sub_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,3,7,2,3,3] vs. [9,8,5,3,2,8,8] [Op:Sub] name: PuTg"
 39. [tf.math.subtract]cannot compute Sub as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Sub] name: WVXv"
 40. [tf.maximum]cannot compute Maximum as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Maximum] name: wwsD"
 41. [tf.maximum]{{function_node __wrapped__Maximum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [2,6,7,1,1,5,1] vs. [7,7,7,7,7,7,7] [Op:Maximum] name: kGZy"
 42. [tf.minimum]cannot compute Minimum as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Minimum] name: uQug"
 43. [tf.minimum]{{function_node __wrapped__Minimum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,7,6,2,7,1,1] vs. [6,1,2,7,2,4,3] [Op:Minimum] name: qPDG"
 44. [tf.multiply]negative dimensions are not allowed"
 45. [tf.multiply]cannot compute Mul as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Mul] name: lBlO"
 46. [tf.multiply]{{function_node __wrapped__Mul_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [6,5,5] vs. [0,6,7,6] [Op:Mul] name: lCKb"
 47. [tf.nn.depth_to_space]Value for attr 'data_format' of "hynU" is not in the list of allowed values: "NHWC", "NCHW", "NCHW_VECT_C"
	; NodeDef: {{node DepthToSpace}}; Op<name=DepthToSpace; signature=input:T -> output:T; attr=T:type; attr=block_size:int,min=2; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW", "NCHW_VECT_C"]> [Op:DepthToSpace] name: twEy"
 48. [tf.nn.depth_to_space]Could not find device for node: {{node DepthToSpace}} = DepthToSpace[T=DT_FLOAT, block_size=5, data_format="NCHW"]
All kernels registered for op DepthToSpace:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='CPU'; T in [DT_VARIANT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_RESOURCE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_STRING]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BOOL]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX128]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_DOUBLE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_FLOAT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BFLOAT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_HALF]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT64]; data_format in ["NHWC"]
  device='GPU'; T in [DT_QINT8]
  device='GPU'; T in [DT_BFLOAT16]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
 [Op:DepthToSpace] name: LmkV"
 49. [tf.nn.gelu]`features.dtype` must be a floating point tensor.Received:features.dtype=<dtype: 'int32'>"
 50. [tf.nn.softmax]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node Softmax}}; Op<name=Softmax; signature=logits:T -> softmax:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:Softmax] name: epwC"
 51. [tf.nn.softmax]`dim` must be in the range [-2, 2) where 2 is the number of dimensions in the input. Received: dim=6"
 52. [tf.nn.space_to_depth]Value for attr 'data_format' of "cFUR" is not in the list of allowed values: "NHWC", "NCHW", "NCHW_VECT_C"
	; NodeDef: {{node SpaceToDepth}}; Op<name=SpaceToDepth; signature=input:T -> output:T; attr=T:type; attr=block_size:int,min=2; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW", "NCHW_VECT_C"]> [Op:SpaceToDepth] name: FlgI"
 53. [tf.nn.space_to_depth]Could not find device for node: {{node SpaceToDepth}} = SpaceToDepth[T=DT_FLOAT, block_size=2, data_format="NCHW"]
All kernels registered for op SpaceToDepth:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='GPU'; T in [DT_UINT8]
  device='GPU'; T in [DT_QINT8]
  device='GPU'; T in [DT_BFLOAT16]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_QINT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_VARIANT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_RESOURCE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_STRING]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BOOL]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX128]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_DOUBLE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_FLOAT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BFLOAT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_HALF]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT64]; data_format in ["NHWC"]
 [Op:SpaceToDepth] name: oGSA"
 54. [tf.nn.space_to_depth]{{function_node __wrapped__SpaceToDepth_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input rank should be 4 instead of 1 [Op:SpaceToDepth] name: frTF"
 55. [tf.nn.space_to_depth]{{function_node __wrapped__SpaceToDepth_device_/job:localhost/replica:0/task:0/device:CPU:0}} Image width 2 and height 7 should be divisible by block_size: 7 [Op:SpaceToDepth] name: oDCt"
 56. [tf.pow]{{function_node __wrapped__Pow_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,4,1,7,2,7,2] vs. [1,1,2,6,1,4,4] [Op:Pow]"
 57. [tf.pow]negative dimensions are not allowed"
 58. [tf.pow]cannot compute Pow as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Pow]"
 59. [tf.raw_ops.AddN]Expected list for 'inputs' argument to 'add_n' Op, not <tf.Tensor: shape=(), dtype=float32, numpy=350280.44>."
 60. [tf.raw_ops.AddV2]cannot compute AddV2 as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:AddV2] name: bhxs"
 61. [tf.raw_ops.AddV2]negative dimensions are not allowed"
 62. [tf.raw_ops.AddV2]{{function_node __wrapped__AddV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [5,3,1,8,9,9,8] vs. [8,8,8,8,8,8,8] [Op:AddV2] name: gTog"
 63. [tf.raw_ops.AdjustContrastv2]{{function_node __wrapped__AdjustContrastv2_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must be at least 3-D, got shape[3,3] [Op:AdjustContrastv2] name: oFwm"
 64. [tf.raw_ops.AdjustHue]Could not find device for node: {{node AdjustHue}} = AdjustHue[T=DT_HALF]
All kernels registered for op AdjustHue:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_HALF]
  device='CPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
 [Op:AdjustHue] name: kyZZ"
 65. [tf.raw_ops.AdjustHue]Value for attr 'T' of int32 is not in the list of allowed values: half, float
	; NodeDef: {{node AdjustHue}}; Op<name=AdjustHue; signature=images:T, delta:float -> output:T; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_FLOAT]> [Op:AdjustHue] name: AXjk"
 66. [tf.raw_ops.AdjustHue]{{function_node __wrapped__AdjustHue_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must have 3 channels but instead has 7 channels. [Op:AdjustHue] name: GCbo"
 67. [tf.raw_ops.AdjustHue]{{function_node __wrapped__AdjustHue_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must be at least 3-D, got shape[1,1] [Op:AdjustHue] name: aYmY"
 68. [tf.raw_ops.AdjustSaturation]{{function_node __wrapped__AdjustSaturation_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must be at least 3-D, got shape[1,1] [Op:AdjustSaturation] name: ozpR"
 69. [tf.raw_ops.All]negative dimensions are not allowed"
 70. [tf.raw_ops.All]Tensor conversion requested dtype bool for Tensor with dtype float32: <tf.Tensor: shape=(3, 4, 9, 2, 7, 7, 8), dtype=float32"
 71. [tf.raw_ops.Any]Value for attr 'Tidx' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node Any}}; Op<name=Any; signature=input:bool, reduction_indices:Tidx -> output:bool; attr=keep_dims:bool,default=false; attr=Tidx:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]> [Op:Any] name: coID"
 72. [tf.raw_ops.Any]cannot compute Any as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:Any] name: DpNH"
 73. [tf.raw_ops.Any]negative dimensions are not allowed"
 74. [tf.raw_ops.ArgMax]{{function_node __wrapped__ArgMax_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected dimension in the range [-2, 2), but got 8 [Op:ArgMax] name: PbbB"
 75. [tf.raw_ops.Asin]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Asin}}; Op<name=Asin; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Asin] name: qnXA"
 76. [tf.raw_ops.Asinh]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Asinh}}; Op<name=Asinh; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Asinh] name: hZPL"
 77. [tf.raw_ops.Atan]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Atan}}; Op<name=Atan; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Atan] name: NbHZ"
 78. [tf.raw_ops.Atan2]cannot compute Atan2 as input #1(zero-based) was expected to be a half tensor but is a double tensor [Op:Atan2] name: OSnm"
 79. [tf.raw_ops.Atan2]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Atan2}}; Op<name=Atan2; signature=y:T, x:T -> z:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Atan2] name: CsAX"
 80. [tf.raw_ops.Atan2]{{function_node __wrapped__Atan2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [8,4,8,4,4] vs. [3,3,3,3,3,3] [Op:Atan2] name: YXpP"
 81. [tf.raw_ops.Atanh]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Atanh}}; Op<name=Atanh; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Atanh] name: oCac"
 82. [tf.raw_ops.BatchMatMulV2]cannot compute BatchMatMulV2 as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:BatchMatMulV2] name: tLxK"
 83. [tf.raw_ops.BatchMatMulV2]{{function_node __wrapped__BatchMatMulV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} In[0] ndims must be >= 2: 0 [Op:BatchMatMulV2] name: poUX"
 84. [tf.raw_ops.BatchMatMulV2]{{function_node __wrapped__BatchMatMulV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} In[0] and In[1] must have compatible batch dimensions: [6,1,9,8,5,1,1] vs. [6,2,2,2,2,2,2] [Op:BatchMatMulV2] name: WkzH"
 85. [tf.raw_ops.BatchMatMulV2]{{function_node __wrapped__BatchMatMulV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Matrix size-incompatible: In[0]: [4,6], In[1]: [4,7] [Op:BatchMatMulV2] name: TzhP"
 86. [tf.raw_ops.BesselI0e]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node BesselI0e}}; Op<name=BesselI0e; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:BesselI0e] name: gTHX"
 87. [tf.raw_ops.BesselI1e]Could not find device for node: {{node BesselI1e}} = BesselI1e[T=DT_BFLOAT16]
All kernels registered for op BesselI1e:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
 [Op:BesselI1e] name: isCD"
 88. [tf.raw_ops.BesselI1e]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node BesselI1e}}; Op<name=BesselI1e; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:BesselI1e] name: CooW"
 89. [tf.raw_ops.Betainc]cannot compute Betainc as input #2(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Betainc] name: GGlo"
 90. [tf.raw_ops.Betainc]Value for attr 'T' of int32 is not in the list of allowed values: float, double
	; NodeDef: {{node Betainc}}; Op<name=Betainc; signature=a:T, b:T, x:T -> z:T; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE]> [Op:Betainc] name: VBEu"
 91. [tf.raw_ops.Betainc]{{function_node __wrapped__Betainc_device_/job:localhost/replica:0/task:0/device:CPU:0}} Shapes of a and b are inconsistent: [9,9] vs. [1,9,9,9] [Op:Betainc] name: elxH"
 92. [tf.raw_ops.BiasAdd]{{function_node __wrapped__BiasAdd_device_/job:localhost/replica:0/task:0/device:CPU:0}} Must provide as many biases as the last dimension of the input tensor: [1] vs. [1,8] [Op:BiasAdd] name: tZoh"
 93. [tf.raw_ops.BiasAdd]cannot compute BiasAdd as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:BiasAdd] name: eQhW"
 94. [tf.raw_ops.BiasAdd]{{function_node __wrapped__BiasAdd_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input tensor must be at least 2D: [] [Op:BiasAdd] name: dRjl"
 95. [tf.raw_ops.BiasAdd]Value for attr 'data_format' of "pZaa" is not in the list of allowed values: "NHWC", "NCHW"
	; NodeDef: {{node BiasAdd}}; Op<name=BiasAdd; signature=value:T, bias:T -> output:T; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 10440210506161272279, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]> [Op:BiasAdd] name: zgOS"
 96. [tf.raw_ops.BiasAddGrad]{{function_node __wrapped__BiasAddGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input tensor must be at least 2D: [] [Op:BiasAddGrad] name: hfwz"
 97. [tf.raw_ops.BiasAddGrad]Value for attr 'data_format' of "llJq" is not in the list of allowed values: "NHWC", "NCHW"
	; NodeDef: {{node BiasAddGrad}}; Op<name=BiasAddGrad; signature=out_backprop:T -> output:T; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 10440210506161272279, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]> [Op:BiasAddGrad] name: oOGq"
 98. [tf.raw_ops.BitwiseAnd]{{function_node __wrapped__BitwiseAnd_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3,3,3,3,3,3] vs. [3,8,5,6,5,5,5] [Op:BitwiseAnd] name: oJgu"
 99. [tf.raw_ops.BitwiseAnd]cannot compute BitwiseAnd as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:BitwiseAnd] name: ATvu"
 100. [tf.raw_ops.BitwiseAnd]Value for attr 'T' of float is not in the list of allowed values: int8, int16, int32, int64, uint8, uint16, uint32, uint64
	; NodeDef: {{node BitwiseAnd}}; Op<name=BitwiseAnd; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]; is_commutative=true> [Op:BitwiseAnd] name: OryT"
 101. [tf.raw_ops.BitwiseOr]{{function_node __wrapped__BitwiseOr_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [6,6,6,6] vs. [6,9,1] [Op:BitwiseOr] name: iVuE"
 102. [tf.raw_ops.BitwiseOr]cannot compute BitwiseOr as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:BitwiseOr] name: wCzz"
 103. [tf.raw_ops.BitwiseOr]Value for attr 'T' of float is not in the list of allowed values: int8, int16, int32, int64, uint8, uint16, uint32, uint64
	; NodeDef: {{node BitwiseOr}}; Op<name=BitwiseOr; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]; is_commutative=true> [Op:BitwiseOr] name: WiDK"
 104. [tf.raw_ops.BitwiseXor]{{function_node __wrapped__BitwiseXor_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,4,4,4,4,4] vs. [7,3,2,3,3] [Op:BitwiseXor] name: Mkcd"
 105. [tf.raw_ops.BitwiseXor]cannot compute BitwiseXor as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:BitwiseXor] name: rKym"
 106. [tf.raw_ops.BitwiseXor]Value for attr 'T' of float is not in the list of allowed values: int8, int16, int32, int64, uint8, uint16, uint32, uint64
	; NodeDef: {{node BitwiseXor}}; Op<name=BitwiseXor; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]; is_commutative=true> [Op:BitwiseXor] name: EOdm"
 107. [tf.raw_ops.BroadcastGradientArgs]Value for attr 'T' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node BroadcastGradientArgs}}; Op<name=BroadcastGradientArgs; signature=s0:T, s1:T -> r0:T, r1:T; attr=T:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]> [Op:BroadcastGradientArgs] name: zxzM"
 108. [tf.raw_ops.BroadcastGradientArgs]cannot compute BroadcastGradientArgs as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:BroadcastGradientArgs] name: ckmN"
 109. [tf.raw_ops.BroadcastGradientArgs]{{function_node __wrapped__BroadcastGradientArgs_device_/job:localhost/replica:0/task:0/device:CPU:0}} In[0] must be a vector.[3,8,8,8,9,8] [Op:BroadcastGradientArgs] name: fwpw"
 110. [tf.raw_ops.BroadcastTo]{{function_node __wrapped__BroadcastTo_device_/job:localhost/replica:0/task:0/device:CPU:0}} Dimension -4 must be >= 0 [Op:BroadcastTo]"
 111. [tf.raw_ops.BroadcastTo]{{function_node __wrapped__BroadcastTo_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3] vs. [9,89,10,16] [Op:BroadcastTo]"
 112. [tf.raw_ops.BroadcastTo]{{function_node __wrapped__BroadcastTo_device_/job:localhost/replica:0/task:0/device:CPU:0}} Rank of input (7) must be no greater than rank of output shape (0). [Op:BroadcastTo]"
 113. [tf.raw_ops.Ceil]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Ceil}}; Op<name=Ceil; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Ceil] name: IeLp"
 114. [tf.raw_ops.Cholesky]Value for attr 'T' of int32 is not in the list of allowed values: double, float, half, complex64, complex128
	; NodeDef: {{node Cholesky}}; Op<name=Cholesky; signature=input:T -> output:T; attr=T:type,allowed=[DT_DOUBLE, DT_FLOAT, DT_HALF, DT_COMPLEX64, DT_COMPLEX128]> [Op:Cholesky] name: qtuq"
 115. [tf.raw_ops.Cholesky]{{function_node __wrapped__Cholesky_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input tensor 0 must have rank >= 2, got 0 [Op:Cholesky] name: EvtN"
 116. [tf.raw_ops.Cholesky]{{function_node __wrapped__Cholesky_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input matrix must be square. [Op:Cholesky] name: TPdt"
 117. [tf.raw_ops.ClipByValue]cannot compute ClipByValue as input #2(zero-based) was expected to be a float tensor but is a int32 tensor [Op:ClipByValue] name: jYbp"
 118. [tf.raw_ops.ClipByValue]{{function_node __wrapped__ClipByValue_device_/job:localhost/replica:0/task:0/device:CPU:0}} clip_value_min and clip_value_max must be either of the same shape as input, or a scalar. input shape: [6,1,9,8,1,2,9]clip_value_min shape: [7,8,2,9,8,7,7]clip_value_max shape: [4,9,9,9,9,9,9] [Op:ClipByValue] name: krkH"
 119. [tf.raw_ops.ComplexAbs]Value for attr 'T' of float is not in the list of allowed values: complex64, complex128
	; NodeDef: {{node ComplexAbs}}; Op<name=ComplexAbs; signature=x:T -> y:Tout; attr=T:type,default=DT_COMPLEX64,allowed=[DT_COMPLEX64, DT_COMPLEX128]; attr=Tout:type,default=DT_FLOAT,allowed=[DT_FLOAT, DT_DOUBLE]> [Op:ComplexAbs] name: xlbG"
 120. [tf.raw_ops.Conj]Value for attr 'T' of float is not in the list of allowed values: complex64, complex128, variant
	; NodeDef: {{node Conj}}; Op<name=Conj; signature=input:T -> output:T; attr=T:type,default=DT_COMPLEX64,allowed=[DT_COMPLEX64, DT_COMPLEX128, DT_VARIANT]> [Op:Conj] name: asRl"
 121. [tf.raw_ops.Cos]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Cos}}; Op<name=Cos; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Cos] name: tVYn"
 122. [tf.raw_ops.Cosh]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Cosh}}; Op<name=Cosh; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Cosh] name: vCuZ"
 123. [tf.raw_ops.Cumprod]{{function_node __wrapped__Cumprod_device_/job:localhost/replica:0/task:0/device:CPU:0}} ScanOp: Expected scan axis in the range [-2, 2), but got 59 [Op:Cumprod] name: oKeu"
 124. [tf.raw_ops.DepthToSpace]{{function_node __wrapped__DepthToSpace_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input depth dimension 7 should be divisible by: 4 [Op:DepthToSpace] name: NkLT"
 125. [tf.raw_ops.DepthToSpace]{{function_node __wrapped__DepthToSpace_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input rank should be: 4 instead of: 3 [Op:DepthToSpace] name: gcev"
 126. [tf.raw_ops.DepthToSpace]Could not find device for node: {{node DepthToSpace}} = DepthToSpace[T=DT_FLOAT, block_size=2, data_format="NCHW_VECT_C"]
All kernels registered for op DepthToSpace:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='CPU'; T in [DT_VARIANT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_RESOURCE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_STRING]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BOOL]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX128]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_DOUBLE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_FLOAT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BFLOAT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_HALF]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT64]; data_format in ["NHWC"]
  device='GPU'; T in [DT_QINT8]
  device='GPU'; T in [DT_BFLOAT16]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
 [Op:DepthToSpace] name: IIit"
 127. [tf.raw_ops.DepthToSpace]negative dimensions are not allowed"
 128. [tf.raw_ops.DepthToSpace]Value for attr 'block_size' of 1 must be at least minimum 2
	; NodeDef: {{node DepthToSpace}}; Op<name=DepthToSpace; signature=input:T -> output:T; attr=T:type; attr=block_size:int,min=2; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW", "NCHW_VECT_C"]> [Op:DepthToSpace] name: HVYc"
 129. [tf.raw_ops.DiagPart]{{function_node __wrapped__DiagPart_device_/job:localhost/replica:0/task:0/device:CPU:0}} The rank of the tensor should be                                          even and positive, got shape [2,2,2] [Op:DiagPart] name: TqDO"
 130. [tf.raw_ops.DiagPart]{{function_node __wrapped__DiagPart_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid shape [3,3,7,3]: dimensions 0 and 2 do not match. [Op:DiagPart] name: VNcg"
 131. [tf.raw_ops.Digamma]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Digamma}}; Op<name=Digamma; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Digamma] name: WxaP"
 132. [tf.raw_ops.DivNoNan]negative dimensions are not allowed"
 133. [tf.raw_ops.DivNoNan]cannot compute DivNoNan as input #1(zero-based) was expected to be a half tensor but is a float tensor [Op:DivNoNan] name: jNGm"
 134. [tf.raw_ops.DivNoNan]Value for attr 'T' of int64 is not in the list of allowed values: half, float, bfloat16, double, complex64, complex128
	; NodeDef: {{node DivNoNan}}; Op<name=DivNoNan; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_HALF, DT_FLOAT, DT_BFLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:DivNoNan] name: Fmxq"
 135. [tf.raw_ops.DivNoNan]{{function_node __wrapped__DivNoNan_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,4,9,8,2,9,8] vs. [8,8,8,8,8,8,8] [Op:DivNoNan] name: aSqQ"
 136. [tf.raw_ops.Elu]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node Elu}}; Op<name=Elu; signature=features:T -> activations:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:Elu] name: MADF"
 137. [tf.raw_ops.EluGrad]cannot compute EluGrad as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:EluGrad] name: PJwy"
 138. [tf.raw_ops.EluGrad]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node EluGrad}}; Op<name=EluGrad; signature=gradients:T, outputs:T -> backprops:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:EluGrad] name: SLIq"
 139. [tf.raw_ops.EluGrad]{{function_node __wrapped__EluGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} Inputs to operation EluGrad of type EluGrad must have the same size and shape.  Input 0: [8,8,8,8] != input 1: [7,3,6,8,3] [Op:EluGrad] name: ytfo"
 140. [tf.raw_ops.Equal]negative dimensions are not allowed"
 141. [tf.raw_ops.Equal]cannot compute Equal as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Equal] name: iIEk"
 142. [tf.raw_ops.Equal]{{function_node __wrapped__Equal_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,8,8,8,8,8,8] vs. [9,9,9,8,8,9] [Op:Equal] name: oOyE"
 143. [tf.raw_ops.Erf]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Erf}}; Op<name=Erf; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Erf] name: KMjt"
 144. [tf.raw_ops.Erfc]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Erfc}}; Op<name=Erfc; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Erfc] name: wqgg"
 145. [tf.raw_ops.Erfinv]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Erfinv}}; Op<name=Erfinv; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Erfinv] name: JgOC"
 146. [tf.raw_ops.Exp]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Exp}}; Op<name=Exp; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Exp] name: mLyQ"
 147. [tf.raw_ops.Expm1]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Expm1}}; Op<name=Expm1; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Expm1] name: kylS"
 148. [tf.raw_ops.Floor]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Floor}}; Op<name=Floor; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Floor] name: nKvO"
 149. [tf.raw_ops.FloorDiv]{{function_node __wrapped__FloorDiv_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [7,8,3,7,8,3,5] vs. [5,5,5,5,5,5,5] [Op:FloorDiv] name: RgSm"
 150. [tf.raw_ops.FloorDiv]cannot compute FloorDiv as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:FloorDiv] name: ZnOE"
 151. [tf.raw_ops.FloorDiv]negative dimensions are not allowed"
 152. [tf.raw_ops.FloorMod]cannot compute FloorMod as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:FloorMod] name: qXQc"
 153. [tf.raw_ops.FloorMod]{{function_node __wrapped__FloorMod_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3,3,3] vs. [5,5] [Op:FloorMod] name: qMxu"
 154. [tf.raw_ops.FusedBatchNorm]Value for attr 'data_format' of "BsrX" is not in the list of allowed values: "NHWC", "NCHW"
	; NodeDef: {{node FusedBatchNorm}}; Op<name=FusedBatchNorm; signature=x:T, scale:T, offset:T, mean:T, variance:T -> y:T, batch_mean:T, batch_variance:T, reserve_space_1:T, reserve_space_2:T; attr=T:type,allowed=[DT_FLOAT]; attr=epsilon:float,default=0.0001; attr=exponential_avg_factor:float,default=1; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]; attr=is_training:bool,default=true> [Op:FusedBatchNorm] name: ByTg"
 155. [tf.raw_ops.FusedBatchNorm]{{function_node __wrapped__FusedBatchNorm_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must be 4 or 5-dimensional[] [Op:FusedBatchNorm] name: ptRW"
 156. [tf.raw_ops.FusedBatchNorm]{{function_node __wrapped__FusedBatchNorm_device_/job:localhost/replica:0/task:0/device:CPU:0}} scale must have the same number of elements as the channels of x, got 7 and 6 [Op:FusedBatchNorm] name: nGlc"
 157. [tf.raw_ops.FusedBatchNorm]cannot compute FusedBatchNorm as input #2(zero-based) was expected to be a float tensor but is a int32 tensor [Op:FusedBatchNorm] name: ztwA"
 159. [tf.raw_ops.GatherNd]{{function_node __wrapped__GatherNd_device_/job:localhost/replica:0/task:0/device:CPU:0}} params must be at least a vector [Op:GatherNd] name: HDDF"
 160. [tf.raw_ops.GatherNd]{{function_node __wrapped__GatherNd_device_/job:localhost/replica:0/task:0/device:CPU:0}} indices[2,3,1,2,2,3] = [-966907, -907408, 989657, -940648] does not index into param shape [4,9,9,9,9,9,9], node name: GatherNd [Op:GatherNd] name: bSmr"
 161. [tf.raw_ops.GatherNd]{{function_node __wrapped__GatherNd_device_/job:localhost/replica:0/task:0/device:CPU:0}} index innermost dimension length must be <= params rank; saw: 3 vs. 1 [Op:GatherNd] name: AFqP"
 162. [tf.raw_ops.Greater]{{function_node __wrapped__Greater_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [5,5] vs. [7,7,7] [Op:Greater] name: MxWo"
 163. [tf.raw_ops.Greater]cannot compute Greater as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Greater] name: Kcqb"
 164. [tf.raw_ops.GreaterEqual]cannot compute GreaterEqual as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:GreaterEqual] name: mAez"
 165. [tf.raw_ops.GreaterEqual]{{function_node __wrapped__GreaterEqual_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [7,5,5,1,5,5] vs. [8,8,8,8] [Op:GreaterEqual] name: hCTT"
 166. [tf.raw_ops.Igamma]negative dimensions are not allowed"
 167. [tf.raw_ops.Igamma]cannot compute Igamma as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:Igamma] name: OFyI"
 168. [tf.raw_ops.Igamma]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Igamma}}; Op<name=Igamma; signature=a:T, x:T -> z:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Igamma] name: ODWA"
 169. [tf.raw_ops.Igamma]{{function_node __wrapped__Igamma_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [9,8,8,8,8,8,8] vs. [9,3,9,8,8,8,8] [Op:Igamma] name: CuEu"
 170. [tf.raw_ops.IgammaGradA]{{function_node __wrapped__IgammaGradA_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [1,6] vs. [9,5,5] [Op:IgammaGradA] name: SdJi"
 171. [tf.raw_ops.IgammaGradA]Value for attr 'T' of int32 is not in the list of allowed values: float, double
	; NodeDef: {{node IgammaGradA}}; Op<name=IgammaGradA; signature=a:T, x:T -> z:T; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE]> [Op:IgammaGradA] name: dGpS"
 172. [tf.raw_ops.IgammaGradA]cannot compute IgammaGradA as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:IgammaGradA] name: sAGM"
 173. [tf.raw_ops.IgammaGradA]negative dimensions are not allowed"
 174. [tf.raw_ops.Imag]Could not find device for node: {{node Imag}} = Imag[T=DT_COMPLEX128, Tout=DT_FLOAT]
All kernels registered for op Imag:
  device='XLA_CPU_JIT'; Tout in [DT_FLOAT, DT_DOUBLE]; T in [DT_COMPLEX64, DT_COMPLEX128]
  device='XLA_GPU_JIT'; Tout in [DT_FLOAT, DT_DOUBLE]; T in [DT_COMPLEX64, DT_COMPLEX128]
  device='GPU'; T in [DT_COMPLEX128]; Tout in [DT_DOUBLE]
  device='GPU'; T in [DT_COMPLEX64]; Tout in [DT_FLOAT]
  device='CPU'; T in [DT_COMPLEX128]; Tout in [DT_DOUBLE]
  device='CPU'; T in [DT_COMPLEX64]; Tout in [DT_FLOAT]
 [Op:Imag] name: zJvk"
 175. [tf.raw_ops.Imag]Value for attr 'T' of float is not in the list of allowed values: complex64, complex128
	; NodeDef: {{node Imag}}; Op<name=Imag; signature=input:T -> output:Tout; attr=T:type,default=DT_COMPLEX64,allowed=[DT_COMPLEX64, DT_COMPLEX128]; attr=Tout:type,default=DT_FLOAT,allowed=[DT_FLOAT, DT_DOUBLE]> [Op:Imag] name: BDyb"
 176. [tf.raw_ops.Invert]Value for attr 'T' of float is not in the list of allowed values: int8, int16, int32, int64, uint8, uint16, uint32, uint64
	; NodeDef: {{node Invert}}; Op<name=Invert; signature=x:T -> y:T; attr=T:type,allowed=[DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]> [Op:Invert] name: Mfrh"
 177. [tf.raw_ops.IsFinite]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node IsFinite}}; Op<name=IsFinite; signature=x:T -> y:bool; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:IsFinite] name: cjqr"
 178. [tf.raw_ops.IsInf]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node IsInf}}; Op<name=IsInf; signature=x:T -> y:bool; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:IsInf] name: GhKA"
 179. [tf.raw_ops.IsNan]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node IsNan}}; Op<name=IsNan; signature=x:T -> y:bool; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:IsNan] name: LyAu"
 180. [tf.raw_ops.LRN]{{function_node __wrapped__LRN_device_/job:localhost/replica:0/task:0/device:CPU:0}} in must be 4-dimensional [Op:LRN] name: cmoW"
 181. [tf.raw_ops.LRN]{{function_node __wrapped__LRN_device_/job:localhost/replica:0/task:0/device:CPU:0}} depth_radius = -3 larger than int max [Op:LRN] name: ABCV"
 182. [tf.raw_ops.LRN]Could not find device for node: {{node LRN}} = LRN[T=DT_BFLOAT16, alpha=5, beta=6, bias=9, depth_radius=0]
All kernels registered for op LRN:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
 [Op:LRN] name: UQaZ"
 183. [tf.raw_ops.LeakyRelu]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node LeakyRelu}}; Op<name=LeakyRelu; signature=features:T -> activations:T; attr=alpha:float,default=0.2; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:LeakyRelu] name: Cyos"
 184. [tf.raw_ops.LeakyReluGrad]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node LeakyReluGrad}}; Op<name=LeakyReluGrad; signature=gradients:T, features:T -> backprops:T; attr=alpha:float,default=0.2; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:LeakyReluGrad] name: ePEA"
 185. [tf.raw_ops.LeakyReluGrad]cannot compute LeakyReluGrad as input #1(zero-based) was expected to be a half tensor but is a float tensor [Op:LeakyReluGrad] name: fGjg"
 186. [tf.raw_ops.LeakyReluGrad]{{function_node __wrapped__LeakyReluGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} Inputs to operation LeakyReluGrad of type LeakyReluGrad must have the same size and shape.  Input 0: [] != input 1: [1,1] [Op:LeakyReluGrad] name: rjEx"
 187. [tf.raw_ops.LeftShift]{{function_node __wrapped__LeftShift_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [5] vs. [2] [Op:LeftShift] name: dVdE"
 188. [tf.raw_ops.LeftShift]cannot compute LeftShift as input #1(zero-based) was expected to be a uint64 tensor but is a float tensor [Op:LeftShift] name: duIe"
 189. [tf.raw_ops.LeftShift]Value for attr 'T' of float is not in the list of allowed values: int8, int16, int32, int64, uint8, uint16, uint32, uint64
	; NodeDef: {{node LeftShift}}; Op<name=LeftShift; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]> [Op:LeftShift] name: pyUc"
 190. [tf.raw_ops.Less]{{function_node __wrapped__Less_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [5,5,5,5,5,5,5] vs. [3,6,9,9,8,9,9] [Op:Less] name: cfAt"
 191. [tf.raw_ops.Less]cannot compute Less as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Less] name: QQIw"
 192. [tf.raw_ops.Less]negative dimensions are not allowed"
 193. [tf.raw_ops.LessEqual]{{function_node __wrapped__LessEqual_device_/job:localhost/replica:0/task:0/device:CPU:0}} Broadcast between [1,7,2,3,1,5,7] and [6,1,2,1,5,1,1] is not supported yet. [Op:LessEqual] name: vOig"
 194. [tf.raw_ops.LessEqual]cannot compute LessEqual as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:LessEqual] name: pHCu"
 195. [tf.raw_ops.LessEqual]negative dimensions are not allowed"
 196. [tf.raw_ops.LessEqual]{{function_node __wrapped__LessEqual_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,4,4,4,4,4,4] vs. [6,1,2,1,3,5,7] [Op:LessEqual] name: DOVZ"
 197. [tf.raw_ops.Lgamma]Could not find device for node: {{node Lgamma}} = Lgamma[T=DT_BFLOAT16]
All kernels registered for op Lgamma:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
 [Op:Lgamma] name: NcAf"
 198. [tf.raw_ops.Lgamma]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Lgamma}}; Op<name=Lgamma; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Lgamma] name: FJcf"
 199. [tf.raw_ops.Log]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Log}}; Op<name=Log; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Log] name: wRQQ"
 200. [tf.raw_ops.Log1p]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Log1p}}; Op<name=Log1p; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Log1p] name: zXPH"
 201. [tf.raw_ops.LogSoftmax]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node LogSoftmax}}; Op<name=LogSoftmax; signature=logits:T -> logsoftmax:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:LogSoftmax] name: joPs"
 202. [tf.raw_ops.LogSoftmax]{{function_node __wrapped__LogSoftmax_device_/job:localhost/replica:0/task:0/device:CPU:0}} logits must have >= 1 dimension, got [] [Op:LogSoftmax] name: FsTm"
 203. [tf.raw_ops.LogicalAnd]{{function_node __wrapped__LogicalAnd_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [6,6,6,6] vs. [8,7,1,8,8] [Op:LogicalAnd] name: cKvI"
 204. [tf.raw_ops.LogicalAnd]cannot compute LogicalAnd as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:LogicalAnd] name: etrS"
 205. [tf.raw_ops.LogicalNot]cannot compute LogicalNot as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:LogicalNot] name: MBTN"
 206. [tf.raw_ops.LogicalOr]cannot compute LogicalOr as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:LogicalOr] name: OxRC"
 207. [tf.raw_ops.LogicalOr]{{function_node __wrapped__LogicalOr_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3,3,3,3] vs. [4,6,7,9,6,6,6] [Op:LogicalOr] name: FFMA"
 208. [tf.raw_ops.LowerBound]{{function_node __wrapped__LowerBound_device_/job:localhost/replica:0/task:0/device:CPU:0}} Leading dim_size of both tensors must match. [Op:LowerBound] name: ICTw"
 209. [tf.raw_ops.LowerBound]{{function_node __wrapped__LowerBound_device_/job:localhost/replica:0/task:0/device:CPU:0}} sorted input argument must be a matrix [Op:LowerBound] name: YdeO"
 210. [tf.raw_ops.MatMul]{{function_node __wrapped__MatMul_device_/job:localhost/replica:0/task:0/device:CPU:0}} Matrix size-incompatible: In[0]: [3,7], In[1]: [1,4] [Op:MatMul] name: kpto"
 211. [tf.raw_ops.MatMul]cannot compute MatMul as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:MatMul] name: Dmip"
 212. [tf.raw_ops.MatMul]{{function_node __wrapped__MatMul_device_/job:localhost/replica:0/task:0/device:CPU:0}} In[0] and In[1] ndims must be == 2: 0 [Op:MatMul] name: Gpva"
 213. [tf.raw_ops.Max]Value for attr 'Tidx' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node Max}}; Op<name=Max; signature=input:T, reduction_indices:Tidx -> output:T; attr=keep_dims:bool,default=false; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 12026250066093653660, DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16]; attr=Tidx:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]> [Op:Max] name: lWCi"
 214. [tf.raw_ops.Max]negative dimensions are not allowed"
 215. [tf.raw_ops.Maximum]cannot compute Maximum as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Maximum] name: Zekm"
 216. [tf.raw_ops.Maximum]{{function_node __wrapped__Maximum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [9,9,9,9,9,9,9] vs. [7,3,1,8,1,2,2] [Op:Maximum] name: ihgx"
 217. [tf.raw_ops.Mean]Value for attr 'Tidx' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node Mean}}; Op<name=Mean; signature=input:T, reduction_indices:Tidx -> output:T; attr=keep_dims:bool,default=false; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 10440210506161272279, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]; attr=Tidx:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]> [Op:Mean] name: NUjZ"
 218. [tf.raw_ops.Mean]negative dimensions are not allowed"
 219. [tf.raw_ops.Min]{{function_node __wrapped__Min_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (9 for input with 5 dimension(s) [Op:Min]"
 220. [tf.raw_ops.Minimum]cannot compute Minimum as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Minimum] name: silH"
 221. [tf.raw_ops.Minimum]{{function_node __wrapped__Minimum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3,3,3] vs. [1,8,2,9,8] [Op:Minimum] name: XdBU"
 222. [tf.raw_ops.Mul]cannot compute Mul as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Mul] name: pmNG"
 223. [tf.raw_ops.Mul]{{function_node __wrapped__Mul_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [5,3,9,3,3,3,3] vs. [4,4,4,4,4,4] [Op:Mul] name: OjGh"
 224. [tf.raw_ops.MulNoNan]{{function_node __wrapped__MulNoNan_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,6,7,8,8,9,8] vs. [8,5,9,5,5,5,5] [Op:MulNoNan] name: Etvj"
 225. [tf.raw_ops.MulNoNan]cannot compute MulNoNan as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:MulNoNan] name: Aeqc"
 226. [tf.raw_ops.Multinomial]cannot compute Multinomial as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Multinomial] name: OrpD"
 227. [tf.raw_ops.Ndtri]Could not find device for node: {{node Ndtri}} = Ndtri[T=DT_HALF]
All kernels registered for op Ndtri:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
 [Op:Ndtri] name: pKEM"
 228. [tf.raw_ops.Ndtri]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Ndtri}}; Op<name=Ndtri; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Ndtri] name: aaiS"
 229. [tf.raw_ops.NextAfter]{{function_node __wrapped__NextAfter_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,4] vs. [1,7,7] [Op:NextAfter] name: SQzZ"
 230. [tf.raw_ops.NextAfter]cannot compute NextAfter as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:NextAfter] name: mcGr"
 231. [tf.raw_ops.NextAfter]Value for attr 'T' of int32 is not in the list of allowed values: double, float
	; NodeDef: {{node NextAfter}}; Op<name=NextAfter; signature=x1:T, x2:T -> output:T; attr=T:type,default=DT_FLOAT,allowed=[DT_DOUBLE, DT_FLOAT]> [Op:NextAfter] name: eYqR"
 232. [tf.raw_ops.NotEqual]cannot compute NotEqual as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:NotEqual] name: PEXb"
 233. [tf.raw_ops.NotEqual]{{function_node __wrapped__NotEqual_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [5,5,5,5,5,5,5] vs. [8,7,6,1,5,7,7] [Op:NotEqual] name: ZRsK"
 234. [tf.raw_ops.OneHot]cannot compute OneHot as input #3(zero-based) was expected to be a float tensor but is a int32 tensor [Op:OneHot] name: fODb"
 235. [tf.raw_ops.OneHot]{{function_node __wrapped__OneHot_device_/job:localhost/replica:0/task:0/device:CPU:0}} depth must be non-negative, got: -1 [Op:OneHot] name: HPmR"
 236. [tf.raw_ops.OneHot]{{function_node __wrapped__OneHot_device_/job:localhost/replica:0/task:0/device:CPU:0}} on_value must be a scalar, but got: [9,8,8,8,5,8,8] [Op:OneHot] name: adob"
 237. [tf.raw_ops.OneHot]{{function_node __wrapped__OneHot_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected axis to be -1 or between [0, 3).  But received: 4 [Op:OneHot] name: xwvZ"
 238. [tf.raw_ops.OneHot]Value for attr 'TI' of float is not in the list of allowed values: uint8, int8, int32, int64
	; NodeDef: {{node OneHot}}; Op<name=OneHot; signature=indices:TI, depth:int32, on_value:T, off_value:T -> output:T; attr=axis:int,default=-1; attr=T:type; attr=TI:type,default=DT_INT64,allowed=[DT_UINT8, DT_INT8, DT_INT32, DT_INT64]> [Op:OneHot] name: oKTT"
 239. [tf.raw_ops.Polygamma]{{function_node __wrapped__Polygamma_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,9] vs. [8,8,8,8] [Op:Polygamma] name: EKDO"
 240. [tf.raw_ops.Polygamma]cannot compute Polygamma as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Polygamma] name: NRKB"
 241. [tf.raw_ops.Polygamma]Value for attr 'T' of int32 is not in the list of allowed values: float, double
	; NodeDef: {{node Polygamma}}; Op<name=Polygamma; signature=a:T, x:T -> z:T; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE]> [Op:Polygamma] name: hgBS"
 242. [tf.raw_ops.Polygamma]negative dimensions are not allowed"
 243. [tf.raw_ops.PopulationCount]Value for attr 'T' of float is not in the list of allowed values: int8, int16, int32, int64, uint8, uint16, uint32, uint64
	; NodeDef: {{node PopulationCount}}; Op<name=PopulationCount; signature=x:T -> y:uint8; attr=T:type,allowed=[DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]> [Op:PopulationCount] name: bVeB"
 244. [tf.raw_ops.Pow]{{function_node __wrapped__Pow_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,4,4,4,4,4,4] vs. [7,8,4,5,6,8,8] [Op:Pow] name: XhGA"
 245. [tf.raw_ops.Pow]cannot compute Pow as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Pow] name: iHrN"
 246. [tf.raw_ops.Prod]{{function_node __wrapped__Prod_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (512323 for input with 7 dimension(s) [Op:Prod] name: iogA"
 247. [tf.raw_ops.Qr]Value for attr 'T' of int32 is not in the list of allowed values: double, float, half, complex64, complex128
	; NodeDef: {{node Qr}}; Op<name=Qr; signature=input:T -> q:T, r:T; attr=full_matrices:bool,default=false; attr=T:type,allowed=[DT_DOUBLE, DT_FLOAT, DT_HALF, DT_COMPLEX64, DT_COMPLEX128]> [Op:Qr] name: IRMt"
 248. [tf.raw_ops.Qr]{{function_node __wrapped__Qr_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input tensor 0 must have rank >= 2, got 0 [Op:Qr] name: UoKw"
 249. [tf.raw_ops.RGBToHSV]{{function_node __wrapped__RGBToHSV_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must have 3 channels but input only has 1 channels. [Op:RGBToHSV] name: xdMB"
 250. [tf.raw_ops.RGBToHSV]{{function_node __wrapped__RGBToHSV_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must be at least 1D[] [Op:RGBToHSV] name: somU"
 251. [tf.raw_ops.RandomGammaGrad]Value for attr 'T' of int32 is not in the list of allowed values: float, double
	; NodeDef: {{node RandomGammaGrad}}; Op<name=RandomGammaGrad; signature=alpha:T, sample:T -> output:T; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE]> [Op:RandomGammaGrad] name: TTMu"
 252. [tf.raw_ops.RandomGammaGrad]cannot compute RandomGammaGrad as input #1(zero-based) was expected to be a float tensor but is a half tensor [Op:RandomGammaGrad] name: Jnaf"
 253. [tf.raw_ops.RandomGammaGrad]{{function_node __wrapped__RandomGammaGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [7,8,7,6] vs. [9,9,9,9,9] [Op:RandomGammaGrad] name: YEYy"
 254. [tf.raw_ops.RealDiv]{{function_node __wrapped__RealDiv_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [9,9,9,9,9,9] vs. [3,9,8,1,8,5,8] [Op:RealDiv] name: dmHk"
 255. [tf.raw_ops.RealDiv]Could not find device for node: {{node RealDiv}} = RealDiv[T=DT_INT32]
All kernels registered for op RealDiv:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='GPU'; T in [DT_COMPLEX128]
  device='GPU'; T in [DT_COMPLEX64]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
 [Op:RealDiv] name: IDQd"
 256. [tf.raw_ops.Reciprocal]Could not find device for node: {{node Reciprocal}} = Reciprocal[T=DT_INT32]
All kernels registered for op Reciprocal:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='GPU'; T in [DT_COMPLEX128]
  device='GPU'; T in [DT_COMPLEX64]
  device='GPU'; T in [DT_INT64]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
 [Op:Reciprocal] name: RMBZ"
 257. [tf.raw_ops.ReciprocalGrad]cannot compute ReciprocalGrad as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:ReciprocalGrad] name: EgAy"
 258. [tf.raw_ops.ReciprocalGrad]{{function_node __wrapped__ReciprocalGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} The two arguments to a cwise op must have same number of elements, got 122500 and 117649 [Op:ReciprocalGrad] name: iNmk"
 259. [tf.raw_ops.Relu6Grad]cannot compute Relu6Grad as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Relu6Grad] name: hqay"
 260. [tf.raw_ops.Relu6Grad]{{function_node __wrapped__Relu6Grad_device_/job:localhost/replica:0/task:0/device:CPU:0}} Inputs to operation Relu6Grad of type Relu6Grad must have the same size and shape.  Input 0: [7,5,3,5,5,5] != input 1: [8,3,5,3] [Op:Relu6Grad] name: ZOLz"
 261. [tf.raw_ops.ResizeBilinearGrad]{{function_node __wrapped__ResizeBilinearGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} original dimensions must be positive [Op:ResizeBilinearGrad] name: WtYJ"
 262. [tf.raw_ops.ResizeBilinearGrad]Value for attr 'T' of int32 is not in the list of allowed values: float, bfloat16, half, double
	; NodeDef: {{node ResizeBilinearGrad}}; Op<name=ResizeBilinearGrad; signature=grads:float, original_image:T -> output:T; attr=T:type,allowed=[DT_FLOAT, DT_BFLOAT16, DT_HALF, DT_DOUBLE]; attr=align_corners:bool,default=false; attr=half_pixel_centers:bool,default=false> [Op:ResizeBilinearGrad] name: ETeW"
 263. [tf.raw_ops.ResizeBilinearGrad]cannot compute ResizeBilinearGrad as input #0(zero-based) was expected to be a float tensor but is a int32 tensor [Op:ResizeBilinearGrad] name: fpeZ"
 264. [tf.raw_ops.ResizeBilinearGrad]{{function_node __wrapped__ResizeBilinearGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} If half_pixel_centers is True, align_corners must be False. [Op:ResizeBilinearGrad] name: AGhH"
 265. [tf.raw_ops.ReverseV2]{{function_node __wrapped__ReverseV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} 'axis'[0] = 418315 is out of valid range [0, 3 [Op:ReverseV2] name: FzFx"
 266. [tf.raw_ops.ReverseV2]{{function_node __wrapped__ReverseV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} 'dims' must be 1-dimension, not 6 [Op:ReverseV2] name: lEag"
 267. [tf.raw_ops.ReverseV2]Value for attr 'Tidx' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node ReverseV2}}; Op<name=ReverseV2; signature=tensor:T, axis:Tidx -> output:T; attr=Tidx:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]; attr=T:type,allowed=[DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_INT32, 5951096766385938332, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_STRING]> [Op:ReverseV2] name: YpCE"
 268. [tf.raw_ops.RightShift]Value for attr 'T' of float is not in the list of allowed values: int8, int16, int32, int64, uint8, uint16, uint32, uint64
	; NodeDef: {{node RightShift}}; Op<name=RightShift; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]> [Op:RightShift] name: PaPF"
 269. [tf.raw_ops.RightShift]cannot compute RightShift as input #1(zero-based) was expected to be a uint32 tensor but is a float tensor [Op:RightShift] name: UTZV"
 270. [tf.raw_ops.RightShift]{{function_node __wrapped__RightShift_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3,3,3] vs. [7,7,8,1,7] [Op:RightShift] name: XtUH"
 271. [tf.raw_ops.Rint]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double
	; NodeDef: {{node Rint}}; Op<name=Rint; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE]> [Op:Rint] name: BTXv"
 272. [tf.raw_ops.RsqrtGrad]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node RsqrtGrad}}; Op<name=RsqrtGrad; signature=y:T, dy:T -> z:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:RsqrtGrad] name: trDP"
 273. [tf.raw_ops.RsqrtGrad]cannot compute RsqrtGrad as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:RsqrtGrad] name: oxfN"
 274. [tf.raw_ops.Select]{{function_node __wrapped__Select_device_/job:localhost/replica:0/task:0/device:CPU:0}} Inputs to operation Select of type Select must have the same size and shape.  Input 0: [9,9,9,9] != input 1: [6,6,6,6,6,6,6] [Op:Select] name: pxJo"
 275. [tf.raw_ops.Select]cannot compute Select as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:Select] name: LGCF"
 276. [tf.raw_ops.SelectV2]negative dimensions are not allowed"
 277. [tf.raw_ops.SelectV2]{{function_node __wrapped__SelectV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} condition [6,9,9,9,9,9,9], then [2,8,5,9,8,8,8], and else [3,4,7,4,4,4,4] must be broadcastable [Op:SelectV2] name: lGPX"
 278. [tf.raw_ops.SelectV2]cannot compute SelectV2 as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:SelectV2] name: bqyi"
 279. [tf.raw_ops.Selu]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node Selu}}; Op<name=Selu; signature=features:T -> activations:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:Selu] name: pAWu"
 280. [tf.raw_ops.SeluGrad]{{function_node __wrapped__SeluGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} Inputs to operation SeluGrad of type SeluGrad must have the same size and shape.  Input 0: [2,9,9,0,1,5,2] != input 1: [4,7,7,7,7,7,7] [Op:SeluGrad] name: snKW"
 281. [tf.raw_ops.SeluGrad]negative dimensions are not allowed"
 282. [tf.raw_ops.SeluGrad]cannot compute SeluGrad as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:SeluGrad] name: tJes"
 283. [tf.raw_ops.SeluGrad]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node SeluGrad}}; Op<name=SeluGrad; signature=gradients:T, outputs:T -> backprops:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:SeluGrad] name: vqyR"
 284. [tf.raw_ops.Sigmoid]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Sigmoid}}; Op<name=Sigmoid; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Sigmoid] name: HuqA"
 285. [tf.raw_ops.SigmoidGrad]{{function_node __wrapped__SigmoidGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} The two arguments to a cwise op must have same number of elements, got 252 and 40 [Op:SigmoidGrad] name: MIPz"
 286. [tf.raw_ops.Sin]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Sin}}; Op<name=Sin; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Sin] name: fezh"
 287. [tf.raw_ops.Sinh]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Sinh}}; Op<name=Sinh; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Sinh] name: iUyJ"
 288. [tf.raw_ops.Slice]Value for attr 'Index' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node Slice}}; Op<name=Slice; signature=input:T, begin:Index, size:Index -> output:T; attr=T:type; attr=Index:type,allowed=[DT_INT32, DT_INT64]> [Op:Slice]"
 289. [tf.raw_ops.Slice]{{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected begin and size arguments to be 1-D tensors of size 3, but got shapes [2] and [0] instead. [Op:Slice]"
 290. [tf.raw_ops.Softmax]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node Softmax}}; Op<name=Softmax; signature=logits:T -> softmax:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:Softmax] name: mwWq"
 291. [tf.raw_ops.Softmax]{{function_node __wrapped__Softmax_device_/job:localhost/replica:0/task:0/device:CPU:0}} logits must have >= 1 dimension, got [] [Op:Softmax] name: cpBy"
 292. [tf.raw_ops.SoftmaxCrossEntropyWithLogits]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node SoftmaxCrossEntropyWithLogits}}; Op<name=SoftmaxCrossEntropyWithLogits; signature=features:T, labels:T -> loss:T, backprop:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:SoftmaxCrossEntropyWithLogits] name: RSse"
 293. [tf.raw_ops.SoftmaxCrossEntropyWithLogits]cannot compute SoftmaxCrossEntropyWithLogits as input #1(zero-based) was expected to be a float tensor but is a half tensor [Op:SoftmaxCrossEntropyWithLogits] name: Mvnf"
 294. [tf.raw_ops.SoftmaxCrossEntropyWithLogits]{{function_node __wrapped__SoftmaxCrossEntropyWithLogits_device_/job:localhost/replica:0/task:0/device:CPU:0}} logits and labels must be broadcastable: logits_size=[9,7] labels_size=[1,2] [Op:SoftmaxCrossEntropyWithLogits] name: tEtf"
 295. [tf.raw_ops.SoftmaxCrossEntropyWithLogits]{{function_node __wrapped__SoftmaxCrossEntropyWithLogits_device_/job:localhost/replica:0/task:0/device:CPU:0}} logits and labels must be either 2-dimensional, or broadcasted to be 2-dimensional [Op:SoftmaxCrossEntropyWithLogits] name: YXSe"
 296. [tf.raw_ops.Softplus]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node Softplus}}; Op<name=Softplus; signature=features:T -> activations:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:Softplus] name: oXfv"
 297. [tf.raw_ops.Softsign]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node Softsign}}; Op<name=Softsign; signature=features:T -> activations:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:Softsign] name: WEOQ"
 298. [tf.raw_ops.SoftsignGrad]cannot compute SoftsignGrad as input #1(zero-based) was expected to be a half tensor but is a float tensor [Op:SoftsignGrad] name: BqKr"
 299. [tf.raw_ops.SoftsignGrad]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node SoftsignGrad}}; Op<name=SoftsignGrad; signature=gradients:T, features:T -> backprops:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:SoftsignGrad] name: Yulk"
 300. [tf.raw_ops.SoftsignGrad]{{function_node __wrapped__SoftsignGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} Inputs to operation SoftsignGrad of type SoftsignGrad must have the same size and shape.  Input 0: [1,1,1,1] != input 1: [3,7,8,7,7] [Op:SoftsignGrad] name: jrXy"
 301. [tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits]{{function_node __wrapped__SparseSoftmaxCrossEntropyWithLogits_device_/job:localhost/replica:0/task:0/device:CPU:0}} Must have at least one class, but got logits shape [0,0] [Op:SparseSoftmaxCrossEntropyWithLogits] name: lZWt"
 302. [tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits]{{function_node __wrapped__SparseSoftmaxCrossEntropyWithLogits_device_/job:localhost/replica:0/task:0/device:CPU:0}} Received a label value of -610134 which is outside the valid range of [0, 1).  Label values: -55761 -610134 385068 [Op:SparseSoftmaxCrossEntropyWithLogits] name: lMEt"
 303. [tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits]{{function_node __wrapped__SparseSoftmaxCrossEntropyWithLogits_device_/job:localhost/replica:0/task:0/device:CPU:0}} logits and labels must have the same first dimension, got logits shape [3,5] and labels shape [7] [Op:SparseSoftmaxCrossEntropyWithLogits] name: bLbH"
 304. [tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits]Value for attr 'Tlabels' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node SparseSoftmaxCrossEntropyWithLogits}}; Op<name=SparseSoftmaxCrossEntropyWithLogits; signature=features:T, labels:Tlabels -> loss:T, backprop:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]; attr=Tlabels:type,default=DT_INT64,allowed=[DT_INT32, DT_INT64]> [Op:SparseSoftmaxCrossEntropyWithLogits] name: THlj"
 305. [tf.raw_ops.Split]Value for attr 'num_split' of 0 must be at least minimum 1
	; NodeDef: {{node Split}}; Op<name=Split; signature=split_dim:int32, value:T -> output:num_split*T; attr=num_split:int,min=1; attr=T:type> [Op:Split] name: PQPs"
 306. [tf.raw_ops.Split]{{function_node __wrapped__Split_num_split_53_device_/job:localhost/replica:0/task:0/device:CPU:0}} Number of ways to split should evenly divide the split dimension, but got split_dim 5 (size = 8) and num_split 53 [Op:Split] name: lncP"
 307. [tf.raw_ops.Split]{{function_node __wrapped__Split_num_split_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} -input rank(-4) <= split_dim < input rank (4), but got 67 [Op:Split] name: jHVg"
 308. [tf.raw_ops.Sqrt]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Sqrt}}; Op<name=Sqrt; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Sqrt] name: yXnD"
 309. [tf.raw_ops.SqrtGrad]{{function_node __wrapped__SqrtGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} The two arguments to a cwise op must have same number of elements, got 81 and 1 [Op:SqrtGrad] name: PqDF"
 310. [tf.raw_ops.SquaredDifference]{{function_node __wrapped__SquaredDifference_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,4] vs. [8,8,8] [Op:SquaredDifference] name: unch"
 311. [tf.raw_ops.SquaredDifference]cannot compute SquaredDifference as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:SquaredDifference] name: yooK"
 312. [tf.raw_ops.Squeeze]{{function_node __wrapped__Squeeze_device_/job:localhost/replica:0/task:0/device:CPU:0}} Tried to squeeze dim index 73 for tensor with 2 dimensions. [Op:Squeeze] name: vCFh"
 313. [tf.raw_ops.StatelessRandomGetKeyCounter]{{function_node __wrapped__StatelessRandomGetKeyCounter_device_/job:localhost/replica:0/task:0/device:CPU:0}} seed must have shape [2], not [0] [Op:StatelessRandomGetKeyCounter]"
 314. [tf.raw_ops.StridedSlice]Value for attr 'Index' of float is not in the list of allowed values: int16, int32, int64
	; NodeDef: {{node StridedSlice}}; Op<name=StridedSlice; signature=input:T, begin:Index, end:Index, strides:Index -> output:T; attr=T:type; attr=Index:type,allowed=[DT_INT16, DT_INT32, DT_INT64]; attr=begin_mask:int,default=0; attr=end_mask:int,default=0; attr=ellipsis_mask:int,default=0; attr=new_axis_mask:int,default=0; attr=shrink_axis_mask:int,default=0> [Op:StridedSlice]"
 315. [tf.raw_ops.StridedSlice]{{function_node __wrapped__StridedSlice_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected begin, end, and strides to be 1D equal size tensors, but got shapes [9], [0], and [0] instead. [Op:StridedSlice]"
 316. [tf.raw_ops.StridedSlice]{{function_node __wrapped__StridedSlice_device_/job:localhost/replica:0/task:0/device:CPU:0}} Multiple ellipses in slice spec not allowed [Op:StridedSlice]"
 317. [tf.raw_ops.StridedSlice]{{function_node __wrapped__StridedSlice_device_/job:localhost/replica:0/task:0/device:CPU:0}} Index out of range using input dim 3; input has only 3 dims [Op:StridedSlice]"
 318. [tf.raw_ops.Sub]cannot compute Sub as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Sub] name: nVJF"
 319. [tf.raw_ops.Sub]{{function_node __wrapped__Sub_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [7,3,7,6,8,8,7] vs. [9,2,2,2,2,2,2] [Op:Sub] name: sVHb"
 320. [tf.raw_ops.Sum]{{function_node __wrapped__Sum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (44 for input with 2 dimension(s) [Op:Sum]"
 321. [tf.raw_ops.Tan]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Tan}}; Op<name=Tan; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Tan] name: VLES"
 322. [tf.raw_ops.Tanh]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Tanh}}; Op<name=Tanh; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Tanh] name: Rbwc"
 323. [tf.raw_ops.TanhGrad]{{function_node __wrapped__TanhGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} The two arguments to a cwise op must have same number of elements, got 288 and 5 [Op:TanhGrad] name: Dfyu"
 324. [tf.raw_ops.Transpose]{{function_node __wrapped__Transpose_device_/job:localhost/replica:0/task:0/device:CPU:0}} transpose expects a vector of size 2. But input(1) is a vector of size 0 [Op:Transpose]"
 325. [tf.raw_ops.Transpose]{{function_node __wrapped__Transpose_device_/job:localhost/replica:0/task:0/device:CPU:0}} -2 is out of range [0 .. 6) [Op:Transpose]"
 326. [tf.raw_ops.TruncateDiv]{{function_node __wrapped__TruncateDiv_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [2] vs. [8] [Op:TruncateDiv] name: daJs"
 327. [tf.raw_ops.TruncateDiv]cannot compute TruncateDiv as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:TruncateDiv] name: fpui"
 328. [tf.raw_ops.TruncateMod]{{function_node __wrapped__TruncateMod_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,1,1,8] vs. [8,8,8,8,8] [Op:TruncateMod] name: ayjw"
 329. [tf.raw_ops.TruncateMod]negative dimensions are not allowed"
 330. [tf.raw_ops.Unpack]{{function_node __wrapped__Unpack_num_4_device_/job:localhost/replica:0/task:0/device:CPU:0}} axis = 21 not in [-5, 5) [Op:Unpack] name: sBVy"
 331. [tf.raw_ops.Unpack]{{function_node __wrapped__Unpack_num_5_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input shape axis 0 must equal 5, got shape [7,7] [Op:Unpack] name: aSaR"
 332. [tf.raw_ops.UpperBound]{{function_node __wrapped__UpperBound_device_/job:localhost/replica:0/task:0/device:CPU:0}} Leading dim_size of both tensors must match. [Op:UpperBound] name: Trcb"
 333. [tf.raw_ops.UpperBound]cannot compute UpperBound as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:UpperBound] name: FRLB"
 334. [tf.raw_ops.UpperBound]{{function_node __wrapped__UpperBound_device_/job:localhost/replica:0/task:0/device:CPU:0}} sorted input argument must be a matrix [Op:UpperBound] name: LzEr"
 335. [tf.raw_ops.Where]{{function_node __wrapped__Where_device_/job:localhost/replica:0/task:0/device:CPU:0}} WhereOp : Unhandled input dimensions: 0 [Op:Where] name: xdWi"
 336. [tf.raw_ops.Xdivy]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double, complex64, complex128
	; NodeDef: {{node Xdivy}}; Op<name=Xdivy; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Xdivy] name: xHdc"
 337. [tf.raw_ops.Xdivy]cannot compute Xdivy as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Xdivy] name: PMZm"
 338. [tf.raw_ops.Xdivy]{{function_node __wrapped__Xdivy_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,3,3,3] vs. [7,3,8,3,3] [Op:Xdivy] name: bCkL"
 339. [tf.raw_ops.Xlog1py]cannot compute Xlog1py as input #1(zero-based) was expected to be a half tensor but is a float tensor [Op:Xlog1py] name: FwhZ"
 340. [tf.raw_ops.Xlog1py]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double, complex64, complex128
	; NodeDef: {{node Xlog1py}}; Op<name=Xlog1py; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Xlog1py] name: yjlT"
 341. [tf.raw_ops.Xlog1py]{{function_node __wrapped__Xlog1py_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [2,7,4,5,5,6,4] vs. [7,7,7,7,7,7,7] [Op:Xlog1py] name: jCbc"
 342. [tf.raw_ops.Xlog1py]negative dimensions are not allowed"
 343. [tf.raw_ops.Xlogy]cannot compute Xlogy as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Xlogy] name: eNbN"
 344. [tf.raw_ops.Xlogy]{{function_node __wrapped__Xlogy_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [3,1,5,1,1,1] vs. [3,1,5,1] [Op:Xlogy] name: TiNd"
 345. [tf.raw_ops.Zeta]{{function_node __wrapped__Zeta_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [4,6,2,2,2,3,1] vs. [4,7,1,1,2,6,3] [Op:Zeta] name: QXtP"
 346. [tf.reverse]Value for attr 'Tidx' of float is not in the list of allowed values: int32, int64
	; NodeDef: {{node ReverseV2}}; Op<name=ReverseV2; signature=tensor:T, axis:Tidx -> output:T; attr=Tidx:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]; attr=T:type,allowed=[DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_INT32, 5951096766385938332, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_STRING]> [Op:ReverseV2] name: evje"
 347. [tf.reverse]{{function_node __wrapped__ReverseV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} 'dims' must be 1-dimension, not 5 [Op:ReverseV2] name: DAXD"
 348. [tf.sigmoid]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Sigmoid}}; Op<name=Sigmoid; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Sigmoid]"
 349. [tf.sqrt]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Sqrt}}; Op<name=Sqrt; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Sqrt] name: QWiN"
 350. [tf.squeeze]{{function_node __wrapped__Squeeze_device_/job:localhost/replica:0/task:0/device:CPU:0}} Tried to squeeze dim index 27 for tensor with 0 dimensions. [Op:Squeeze] name: Llqa"
 351. [tf.tan]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Tan}}; Op<name=Tan; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Tan] name: WdYU"
 352. [tf.transpose]{{function_node __wrapped__Transpose_device_/job:localhost/replica:0/task:0/device:CPU:0}} 0 is missing from {2,2,2,2,2,2,2}. [Op:Transpose]"
 353. [tf.transpose]{{function_node __wrapped__Transpose_device_/job:localhost/replica:0/task:0/device:CPU:0}} transpose expects a vector of size 7. But input(1) is a vector of size 0 [Op:Transpose]"
 354. [tf.where]negative dimensions are not allowed"
 355. [tf.where]{{function_node __wrapped__SelectV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} condition [], then [5,4,7,5,2,5], and else [1,5,2,5,5,5,5] must be broadcastable [Op:SelectV2] name: WrQG"
 356. [tf.where]cannot compute SelectV2 as input #0(zero-based) was expected to be a bool tensor but is a float tensor [Op:SelectV2] name: rQpT"
 0. [tf.add]cannot compute AddV2 as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:AddV2]"
 1. [tf.add]negative dimensions are not allowed"
 2. [tf.experimental.numpy.log2]EagerTensor object has no attribute 'astype'. 
        If you are looking for numpy-related methods, please run the following:
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()
      "
 3. [tf.gather]{{function_node __wrapped__GatherV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} params must be at least 1 dimensional [Op:GatherV2] name: YPix"
 4. [tf.linalg.cholesky]Value for attr 'T' of int32 is not in the list of allowed values: double, float, half, complex64, complex128
	; NodeDef: {{node Cholesky}}; Op<name=Cholesky; signature=input:T -> output:T; attr=T:type,allowed=[DT_DOUBLE, DT_FLOAT, DT_HALF, DT_COMPLEX64, DT_COMPLEX128]> [Op:Cholesky] name: DOKo"
 5. [tf.linalg.eigh]Value for attr 'T' of int32 is not in the list of allowed values: double, float, half, complex64, complex128
	; NodeDef: {{node SelfAdjointEigV2}}; Op<name=SelfAdjointEigV2; signature=input:T -> e:T, v:T; attr=compute_v:bool,default=true; attr=T:type,allowed=[DT_DOUBLE, DT_FLOAT, DT_HALF, DT_COMPLEX64, DT_COMPLEX128]> [Op:SelfAdjointEigV2] name: lHaD"
 6. [tf.math.logical_xor]negative dimensions are not allowed"
 7. [tf.nn.depth_to_space]{{function_node __wrapped__DepthToSpace_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input depth dimension 4 should be divisible by: 16 [Op:DepthToSpace] name: UTDh"
 8. [tf.pow]{{function_node __wrapped__Pow_device_/job:localhost/replica:0/task:0/device:CPU:0}} Integers to negative integer powers are not allowed [Op:Pow]"
 9. [tf.raw_ops.Acos]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Acos}}; Op<name=Acos; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Acos] name: TLdS"
 10. [tf.raw_ops.Acosh]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Acosh}}; Op<name=Acosh; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Acosh] name: YKzs"
 11. [tf.raw_ops.AddN]OpKernel 'AddN' has constraint on attr 'T' not in NodeDef '[N=0]', KernelDef: 'op: "AddN" device_type: "CPU" constraint { name: "T" allowed_values { list { type: DT_VARIANT } } }' [Op:AddN] name: CZOK"
 12. [tf.raw_ops.AdjustContrastv2]Value for attr 'T' of int32 is not in the list of allowed values: half, float
	; NodeDef: {{node AdjustContrastv2}}; Op<name=AdjustContrastv2; signature=images:T, contrast_factor:float -> output:T; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_FLOAT]> [Op:AdjustContrastv2] name: pQjM"
 13. [tf.raw_ops.AdjustSaturation]{{function_node __wrapped__AdjustSaturation_device_/job:localhost/replica:0/task:0/device:CPU:0}} input must have 3 channels but instead has 1 channels. [Op:AdjustSaturation] name: dhUO"
 14. [tf.raw_ops.AdjustSaturation]Value for attr 'T' of int32 is not in the list of allowed values: half, float
	; NodeDef: {{node AdjustSaturation}}; Op<name=AdjustSaturation; signature=images:T, scale:float -> output:T; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_FLOAT]> [Op:AdjustSaturation] name: mTte"
 15. [tf.raw_ops.Any]{{function_node __wrapped__Any_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (226975 for input with 3 dimension(s) [Op:Any] name: XbDn"
 16. [tf.raw_ops.ArgMax]{{function_node __wrapped__ArgMax_device_/job:localhost/replica:0/task:0/device:CPU:0}} Reduction axis -4 is empty in shape [7,2,5,0,2,2,2] [Op:ArgMax] name: guEj"
 17. [tf.raw_ops.BesselI0e]Could not find device for node: {{node BesselI0e}} = BesselI0e[T=DT_BFLOAT16]
All kernels registered for op BesselI0e:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
 [Op:BesselI0e] name: Wrcr"
 18. [tf.raw_ops.Betainc]negative dimensions are not allowed"
 19. [tf.raw_ops.BroadcastGradientArgs]{{function_node __wrapped__BroadcastGradientArgs_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [226975,850569] vs. [541030,375367,38339,155074] [Op:BroadcastGradientArgs] name: Txij"
 20. [tf.raw_ops.BroadcastTo]{{function_node __wrapped__BroadcastTo_device_/job:localhost/replica:0/task:0/device:CPU:0}} Unable to broadcast tensor of shape [5,1,1,1,1,1,1] to tensor of shape [1,2,2,2,2,2,2] [Op:BroadcastTo]"
 21. [tf.raw_ops.Bucketize]{{function_node __wrapped__Bucketize_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected sorted boundaries [Op:Bucketize] name: vUzp"
 22. [tf.raw_ops.ComplexAbs]{{function_node __wrapped__ComplexAbs_device_/job:localhost/replica:0/task:0/device:CPU:0}} Signature mismatch, have: complex128->float expected: complex128->double [Op:ComplexAbs] name: ZhEX"
 23. [tf.raw_ops.Digamma]Could not find device for node: {{node Digamma}} = Digamma[T=DT_BFLOAT16]
All kernels registered for op Digamma:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
 [Op:Digamma] name: UdHR"
 24. [tf.raw_ops.Erfinv]Could not find device for node: {{node Erfinv}} = Erfinv[T=DT_BFLOAT16]
All kernels registered for op Erfinv:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
 [Op:Erfinv] name: RanI"
 25. [tf.raw_ops.Igamma]Could not find device for node: {{node Igamma}} = Igamma[T=DT_HALF]
All kernels registered for op Igamma:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
 [Op:Igamma] name: dYSO"
 26. [tf.raw_ops.LRN]Value for attr 'T' of double is not in the list of allowed values: half, bfloat16, float
	; NodeDef: {{node LRN}}; Op<name=LRN; signature=input:T -> output:T; attr=depth_radius:int,default=5; attr=bias:float,default=1; attr=alpha:float,default=1; attr=beta:float,default=0.5; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT]> [Op:LRN]"
 27. [tf.raw_ops.LowerBound]cannot compute LowerBound as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:LowerBound] name: LeCc"
 28. [tf.raw_ops.Max]{{function_node __wrapped__Max_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (226975 for input with 0 dimension(s) [Op:Max] name: XLte"
 29. [tf.raw_ops.Mean]{{function_node __wrapped__Mean_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction dimension (226975 for input with 7 dimension(s) [Op:Mean] name: pajq"
 30. [tf.raw_ops.Min]{{function_node __wrapped__Min_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction arguments: Axes contains duplicate dimension: 6 [Op:Min]"
 31. [tf.raw_ops.Minimum]negative dimensions are not allowed"
 32. [tf.raw_ops.Multinomial]{{function_node __wrapped__Multinomial_device_/job:localhost/replica:0/task:0/device:CPU:0}} logits should be a matrix, got shape [] [Op:Multinomial] name: orir"
 33. [tf.raw_ops.Multinomial]Could not find device for node: {{node Multinomial}} = Multinomial[T=DT_INT32, output_dtype=DT_INT64, seed=-4, seed2=3]
All kernels registered for op Multinomial:
  device='XLA_CPU_JIT'; output_dtype in [DT_INT32, DT_INT64]; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64, DT_BFLOAT16, DT_UINT16, DT_HALF, DT_UINT32, DT_UINT64]
  device='XLA_GPU_JIT'; output_dtype in [DT_INT32, DT_INT64]; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64, DT_BFLOAT16, DT_UINT16, DT_HALF, DT_UINT32, DT_UINT64]
  device='GPU'; T in [DT_DOUBLE]; output_dtype in [DT_INT64]
  device='GPU'; T in [DT_DOUBLE]; output_dtype in [DT_INT32]
  device='GPU'; T in [DT_FLOAT]; output_dtype in [DT_INT64]
  device='GPU'; T in [DT_FLOAT]; output_dtype in [DT_INT32]
  device='GPU'; T in [DT_HALF]; output_dtype in [DT_INT64]
  device='GPU'; T in [DT_HALF]; output_dtype in [DT_INT32]
  device='CPU'; T in [DT_DOUBLE]; output_dtype in [DT_INT64]
  device='CPU'; T in [DT_DOUBLE]; output_dtype in [DT_INT32]
  device='CPU'; T in [DT_FLOAT]; output_dtype in [DT_INT64]
  device='CPU'; T in [DT_FLOAT]; output_dtype in [DT_INT32]
  device='CPU'; T in [DT_HALF]; output_dtype in [DT_INT64]
  device='CPU'; T in [DT_HALF]; output_dtype in [DT_INT32]
 [Op:Multinomial] name: kYml"
 34. [tf.raw_ops.NextAfter]negative dimensions are not allowed"
 35. [tf.raw_ops.Pow]{{function_node __wrapped__Pow_device_/job:localhost/replica:0/task:0/device:CPU:0}} Integers to negative integer powers are not allowed [Op:Pow] name: SgWR"
 36. [tf.raw_ops.RGBToHSV]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double
	; NodeDef: {{node RGBToHSV}}; Op<name=RGBToHSV; signature=images:T -> output:T; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:RGBToHSV] name: PdRe"
 37. [tf.raw_ops.RealDiv]negative dimensions are not allowed"
 38. [tf.raw_ops.ReciprocalGrad]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node ReciprocalGrad}}; Op<name=ReciprocalGrad; signature=y:T, dy:T -> z:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:ReciprocalGrad] name: Cblz"
 39. [tf.raw_ops.Reshape]{{function_node __wrapped__Reshape_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input to reshape is a tensor with 42 values, but the requested shape has 1 [Op:Reshape]"
 40. [tf.raw_ops.Rint]Could not find device for node: {{node Rint}} = Rint[T=DT_BFLOAT16]
All kernels registered for op Rint:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_BFLOAT16, DT_HALF]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
 [Op:Rint] name: vVxK"
 41. [tf.raw_ops.Rsqrt]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node Rsqrt}}; Op<name=Rsqrt; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Rsqrt] name: VGkA"
 42. [tf.raw_ops.RsqrtGrad]{{function_node __wrapped__RsqrtGrad_device_/job:localhost/replica:0/task:0/device:CPU:0}} The two arguments to a cwise op must have same number of elements, got 3 and 1 [Op:RsqrtGrad] name: DzEM"
 43. [tf.raw_ops.SigmoidGrad]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node SigmoidGrad}}; Op<name=SigmoidGrad; signature=y:T, dy:T -> z:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:SigmoidGrad] name: DaHI"
 44. [tf.raw_ops.SigmoidGrad]cannot compute SigmoidGrad as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:SigmoidGrad] name: hHfo"
 45. [tf.raw_ops.Slice]{{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:CPU:0}} Expected size[0] in [0, 3], but got 6 [Op:Slice]"
 46. [tf.raw_ops.Split]negative dimensions are not allowed"
 47. [tf.raw_ops.SqrtGrad]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node SqrtGrad}}; Op<name=SqrtGrad; signature=y:T, dy:T -> z:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:SqrtGrad] name: pWPz"
 48. [tf.raw_ops.SqrtGrad]cannot compute SqrtGrad as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:SqrtGrad] name: hGNx"
 49. [tf.raw_ops.StridedSlice]{{function_node __wrapped__StridedSlice_device_/job:localhost/replica:0/task:0/device:CPU:0}} only stride 1 allowed on non-range indexing. [Op:StridedSlice]"
 50. [tf.raw_ops.StridedSlice]{{function_node __wrapped__StridedSlice_device_/job:localhost/replica:0/task:0/device:CPU:0}} slice index 7 of dimension 0 out of bounds. [Op:StridedSlice]"
 51. [tf.raw_ops.StridedSlice]{{function_node __wrapped__StridedSlice_device_/job:localhost/replica:0/task:0/device:CPU:0}} strides[1] must be non-zero [Op:StridedSlice]"
 52. [tf.raw_ops.Sum]{{function_node __wrapped__Sum_device_/job:localhost/replica:0/task:0/device:CPU:0}} Invalid reduction arguments: Axes contains duplicate dimension: 5 [Op:Sum]"
 53. [tf.raw_ops.TanhGrad]Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
	; NodeDef: {{node TanhGrad}}; Op<name=TanhGrad; signature=y:T, dy:T -> z:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:TanhGrad] name: hutd"
 54. [tf.raw_ops.TanhGrad]cannot compute TanhGrad as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:TanhGrad] name: kymn"
 55. [tf.raw_ops.Transpose]{{function_node __wrapped__Transpose_device_/job:localhost/replica:0/task:0/device:CPU:0}} 2 is missing from {4,1,0,1,1,1}. [Op:Transpose]"
 56. [tf.raw_ops.TruncateMod]cannot compute TruncateMod as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:TruncateMod] name: wpvC"
 57. [tf.raw_ops.Unpack]Value for attr 'num' of -2 must be at least minimum 0
	; NodeDef: {{node Unpack}}; Op<name=Unpack; signature=value:T -> output:num*T; attr=num:int,min=0; attr=T:type; attr=axis:int,default=0> [Op:Unpack]"
 58. [tf.raw_ops.Xlogy]Value for attr 'T' of int32 is not in the list of allowed values: half, bfloat16, float, double, complex64, complex128
	; NodeDef: {{node Xlogy}}; Op<name=Xlogy; signature=x:T, y:T -> z:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Xlogy] name: FhDl"
 59. [tf.raw_ops.Xlogy]negative dimensions are not allowed"
 60. [tf.raw_ops.Zeta]negative dimensions are not allowed"
 61. [tf.raw_ops.Zeta]cannot compute Zeta as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:Zeta] name: kylb"
 62. [tf.raw_ops.Zeta]Value for attr 'T' of int32 is not in the list of allowed values: float, double
	; NodeDef: {{node Zeta}}; Op<name=Zeta; signature=x:T, q:T -> z:T; attr=T:type,allowed=[DT_FLOAT, DT_DOUBLE]> [Op:Zeta] name: JXpR"
 63. [tf.reverse]{{function_node __wrapped__ReverseV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} 'axis'[0] = -654459 is out of valid range [0, 1 [Op:ReverseV2] name: DbkZ"
 64. [tf.squeeze]{{function_node __wrapped__Squeeze_device_/job:localhost/replica:0/task:0/device:CPU:0}} Can not squeeze dim[5], expected a dimension of 1, got 5 [Op:Squeeze] name: DCvm"

## Bad Error Message
1.  [too verbose][tf.raw_ops.GatherNd]Could not find device for node: {{node GatherNd}} = GatherNd[Tindices=DT_INT16, Tparams=DT_FLOAT]
All kernels registered for op GatherNd:
  device='XLA_CPU_JIT'; Tindices in [DT_INT32, DT_INT16, DT_INT64]; Tparams in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='XLA_GPU_JIT'; Tindices in [DT_INT32, DT_INT16, DT_INT64]; Tparams in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 930109355527764061, DT_HALF, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]
  device='GPU'; Tparams in [DT_COMPLEX128]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_COMPLEX128]; Tindices in [DT_INT32]
  device='GPU'; Tparams in [DT_COMPLEX64]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_COMPLEX64]; Tindices in [DT_INT32]
  device='GPU'; Tparams in [DT_DOUBLE]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_DOUBLE]; Tindices in [DT_INT32]
  device='GPU'; Tparams in [DT_FLOAT]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_FLOAT]; Tindices in [DT_INT32]
  device='GPU'; Tparams in [DT_BFLOAT16]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_BFLOAT16]; Tindices in [DT_INT32]
  device='GPU'; Tparams in [DT_HALF]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_HALF]; Tindices in [DT_INT32]
  device='GPU'; Tparams in [DT_INT64]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_INT64]; Tindices in [DT_INT32]
  device='GPU'; Tparams in [DT_INT32]; Tindices in [DT_INT64]
  device='GPU'; Tparams in [DT_INT32]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_QINT32]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_QINT32]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_QUINT8]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_QUINT8]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_QINT8]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_QINT8]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_VARIANT]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_VARIANT]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_RESOURCE]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_RESOURCE]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_STRING]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_STRING]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_BOOL]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_BOOL]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_COMPLEX128]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_COMPLEX128]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_COMPLEX64]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_COMPLEX64]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_DOUBLE]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_DOUBLE]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_FLOAT]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_FLOAT]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_BFLOAT16]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_BFLOAT16]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_HALF]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_HALF]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_INT32]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_INT32]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_INT8]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_INT8]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_UINT8]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_UINT8]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_INT16]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_INT16]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_UINT16]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_UINT16]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_UINT32]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_UINT32]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_INT64]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_INT64]; Tindices in [DT_INT32]
  device='CPU'; Tparams in [DT_UINT64]; Tindices in [DT_INT64]
  device='CPU'; Tparams in [DT_UINT64]; Tindices in [DT_INT32]
 [Op:GatherNd] name: yRiT"