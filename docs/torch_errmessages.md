## Good Error Messages

 0. [torch.Tensor.and]bitwise_and_cpu not implemented for 'Float'
 1. [torch.Tensor.iand]output with shape [] doesn't match the broadcast shape [5]
 2. [torch.Tensor.iand]The size of tensor a (7) must match the size of tensor b (0) at non-singleton dimension 1
 4. [torch.Tensor.iand]result type Float can't be cast to the desired output type Int
 5. [torch.Tensor.ior]result type Float can't be cast to the desired output type Int
 7. [torch.Tensor.ior]The size of tensor a (6) must match the size of tensor b (8) at non-singleton dimension 3
 8. [torch.Tensor.ixor]output with shape [1] doesn't match the broadcast shape [1, 8, 8, 9, 8, 5]
 373. [torch._C._fft.fft_irfft]Invalid number of data points (0) specified
 374. [torch._C._fft.fft_irfft]Dimension specified as 78 but tensor has no dimensions
 375. [torch._C._fft.fft_irfft]Invalid normalization mode: CiVX
 376. [torch._C._fft.fft_irfft]Trying to resize storage that is not resizable
 377. [torch._C._fft.fft_irfft]irfft expects a floating point output tensor, but got Int
 378. [torch._C._fft.fft_irfft2]irfftn must transform at least one axis
 11. [torch.Tensor.ixor]The size of tensor a (9) must match the size of tensor b (6) at non-singleton dimension 6
 12. [torch.Tensor.lshift]The size of tensor a (9) must match the size of tensor b (3) at non-singleton dimension 2
 13. [torch.Tensor.lshift]lshift_cpu not implemented for 'Float'
 14. [torch.Tensor.or]The size of tensor a (7) must match the size of tensor b (4) at non-singleton dimension 0
 17. [torch.Tensor.rshift]The size of tensor a (5) must match the size of tensor b (8) at non-singleton dimension 6
 21. [torch.Tensor.xor]The size of tensor a (9) must match the size of tensor b (6) at non-singleton dimension 5
 23. [torch.Tensor.acos_]result type Float can't be cast to the desired output type Int
 24. [torch.Tensor.add]The size of tensor a (2) must match the size of tensor b (4) at non-singleton dimension 6
 25. [torch.Tensor.add_]output with shape [1] doesn't match the broadcast shape [1, 9, 8, 2, 9]
 26. [torch.Tensor.addbmm]Dimension out of range (expected to be in range of [-1, 0], but got 1)
 27. [torch.Tensor.addbmm]The expanded size of the tensor (4) must match the existing size (5) at non-singleton dimension 1.  Target sizes: [2, 4].  Tensor sizes: [5]
 28. [torch.Tensor.addbmm]Dimension specified as 1 but tensor has no dimensions
 29. [torch.Tensor.addbmm]batch2 must be a 3D tensor
 30. [torch.Tensor.addbmm]Incompatible matrix sizes for bmm (6x8 and 5x2)
 31. [torch.Tensor.addbmm]batch1 and batch2 must have same number of batches, got 9 and 8
 32. [torch.Tensor.addcdiv]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 5
 33. [torch.Tensor.addcdiv_]Integer division with addcdiv is no longer supported, and in a future  release addcdiv will perform a true division of tensor1 and tensor2. The historic addcdiv behavior can be implemented as (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for integer inputs and as (input + value * tensor1 / tensor2) for float inputs. The future addcdiv behavior is just the latter implementation: (input + value * tensor1 / tensor2), for all dtypes.
 34. [torch.Tensor.addcdiv_]The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 0
 35. [torch.Tensor.addcdiv_]output with shape [1, 9, 1, 5] doesn't match the broadcast shape [1, 9, 9, 5]
 36. [torch.Tensor.addcmul]The size of tensor a (4) must match the size of tensor b (6) at non-singleton dimension 5
 37. [torch.Tensor.addcmul_]result type Float can't be cast to the desired output type Int
 38. [torch.Tensor.addcmul_]The size of tensor a (9) must match the size of tensor b (4) at non-singleton dimension 6
 39. [torch.Tensor.addcmul_]output with shape [1] doesn't match the broadcast shape [1, 6, 6, 6]
 40. [torch.Tensor.addmm]The expanded size of the tensor (9) must match the existing size (5) at non-singleton dimension 1.  Target sizes: [7, 9].  Tensor sizes: [8, 5]
 41. [torch.Tensor.addmm]expand(torch.FloatTensor{[8, 2, 9, 2, 1, 5, 4]}, size=[2, 5]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (7)
 42. [torch.Tensor.addmm]mat1 and mat2 shapes cannot be multiplied (5x9 and 7x5)
 43. [torch.Tensor.addmm]self and mat2 must have the same dtype, but got Int and Float
 44. [torch.Tensor.addmm]mat1 must be a matrix, got 7-D tensor
 45. [torch.Tensor.addmm]mat1 and mat2 must have the same dtype
 46. [torch.Tensor.addmm_]mat1 must be a matrix, got 4-D tensor
 47. [torch.Tensor.addmm_]self and mat2 must have the same dtype, but got Float and Int
 48. [torch.Tensor.addmm_]mat1 and mat2 shapes cannot be multiplied (3x4 and 8x5)
 49. [torch.Tensor.addmm_]Bad in-place call: input tensor size [8, 6, 7, 8, 2, 7, 8] and output tensor size [9, 0] should match
 50. [torch.Tensor.addmm_]mat1 and mat2 must have the same dtype
 51. [torch.Tensor.addmv]vector + matrix @ vector expected, got 1, 1, 1
 52. [torch.Tensor.addmv_]Bad in-place call: input tensor dtype int and output tensor dtype float should match
 53. [torch.Tensor.addmv_]Bad in-place call: input tensor size [1] and output tensor size [2] should match
 54. [torch.Tensor.addmv_]Dimension specified as 0 but tensor has no dimensions
 55. [torch.Tensor.addmv_]size mismatch, got input (4), mat (4x7), vec (3)
 56. [torch.Tensor.addmv_]vector + matrix @ vector expected, got 7, 0, 7
 57. [torch.Tensor.addr]addr: Expected 1-D argument vec1, but got 2-D
 58. [torch.Tensor.addr]The expanded size of the tensor (7) must match the existing size (4) at non-singleton dimension 1.  Target sizes: [6, 7].  Tensor sizes: [4]
 59. [torch.Tensor.adjoint]tensor.adjoint() is only supported on matrices or batches of matrices. Got 1-D tensor.
 60. [torch.Tensor.amax]Dimension out of range (expected to be in range of [-1, 0], but got 5)
 61. [torch.Tensor.amin]Dimension out of range (expected to be in range of [-1, 0], but got 6)
 62. [torch.Tensor.aminmax]Dimension out of range (expected to be in range of [-1, 0], but got 3)
 63. [torch.Tensor.argmax]Dimension out of range (expected to be in range of [-1, 0], but got -4)
 64. [torch.Tensor.argmax]argmax(): Expected reduction dim 0 to have non-zero size.
 65. [torch.Tensor.argmin]Dimension out of range (expected to be in range of [-1, 0], but got 5)
 66. [torch.Tensor.argmin]argmin(): Expected reduction dim to be specified for input.numel() == 0.
 67. [torch.Tensor.argmin]argmin(): Expected reduction dim 0 to have non-zero size.
 68. [torch.Tensor.argsort]Dimension out of range (expected to be in range of [-1, 0], but got -4)
 69. [torch.Tensor.as_strided]Storage size calculation overflowed with sizes=[-3, -2] and strides=[9, 7]
 70. [torch.Tensor.as_strided]as_strided: Negative strides are not supported at the moment, got strides: [9, 7, 3, -3, -4, 6, -1]
 71. [torch.Tensor.as_strided]mismatch in length of strides and shape
 72. [torch.Tensor.as_strided_]mismatch in length of strides and shape
 73. [torch.Tensor.as_strided_]numel: integer multiplication overflow
 74. [torch.Tensor.as_strided_]as_strided: Negative strides are not supported at the moment, got strides: [-1]
 75. [torch.Tensor.atan2]The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 5
 76. [torch.Tensor.atan2_]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 5
 77. [torch.Tensor.atan2_]Too large tensor shape: shape = [6, 9, 9, 9, 9, 9, 9, 9, 9]
 78. [torch.Tensor.atan2_]output with shape [3, 5, 1, 5, 5, 5] doesn't match the broadcast shape [3, 5, 3, 5, 5, 5]
 79. [torch.Tensor.atan2_]result type Float can't be cast to the desired output type Int
 80. [torch.Tensor.bitwise_and]bitwise_and_cpu not implemented for 'Float'
 81. [torch.Tensor.bitwise_and]The size of tensor a (9) must match the size of tensor b (6) at non-singleton dimension 1
 82. [torch.Tensor.bitwise_left_shift]lshift_cpu not implemented for 'Float'
 83. [torch.Tensor.bitwise_left_shift]The size of tensor a (3) must match the size of tensor b (6) at non-singleton dimension 0
 84. [torch.Tensor.bitwise_not]bitwise_not_cpu not implemented for 'Float'
 85. [torch.Tensor.bitwise_not_]bitwise_not_cpu not implemented for 'Float'
 86. [torch.Tensor.bitwise_or]Too large tensor shape: shape = [5, 8, 9, 9, 9, 9, 9, 9, 9]
 87. [torch.Tensor.bitwise_or]bitwise_or_cpu not implemented for 'Float'
 88. [torch.Tensor.bitwise_or]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 5
 89. [torch.Tensor.bitwise_xor]bitwise_xor_cpu not implemented for 'Float'
 90. [torch.Tensor.bitwise_xor]The size of tensor a (9) must match the size of tensor b (3) at non-singleton dimension 4
 91. [torch.Tensor.bmm]batch1 must be a 3D tensor
 92. [torch.Tensor.bmm]expected scalar type Int but found Float
 93. [torch.Tensor.bmm]Expected size for first two dimensions of batch2 tensor to be: [2, 8] but got: [1, 5].
 94. [torch.Tensor.bmm]bmm not implemented for 'Bool'
 96. [torch.Tensor.broadcast_to]numel: integer multiplication overflow
 97. [torch.Tensor.broadcast_to]expand(torch.FloatTensor{[8, 4, 9, 4, 4, 4, 4, 4, 4]}, size=[]): the number of sizes provided (0) must be greater or equal to the number of dimensions in the tensor (9)
 98. [torch.Tensor.cholesky_solve]Too large tensor shape: shape = [8, 8, 9, 8, 8, 8, 8, 8, 8]
 99. [torch.Tensor.cholesky_solve]Incompatible matrix sizes for cholesky_solve: each A matrix is 1 by 1 but each b matrix is 9 by 9
 100. [torch.Tensor.cholesky_solve]u should have at least 2 dimensions, but has 0 dimensions instead
 101. [torch.Tensor.chunk]chunk expects at least a 1-dimensional tensor
 102. [torch.Tensor.chunk]chunk expects `chunks` to be greater than 0, got: 0
 103. [torch.Tensor.chunk]Dimension out of range (expected to be in range of [-1, 0], but got 82)
 104. [torch.Tensor.clamp]negative dimensions are not allowed
 105. [torch.Tensor.copysign]The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 2
 106. [torch.Tensor.count_nonzero]Dimension out of range (expected to be in range of [-4, 3], but got 5)
 107. [torch.Tensor.cov]cov(): expected fweights to have integral dtype but got fweights with Float dtype
 108. [torch.Tensor.cov]cov(): expected aweights to have one or fewer dimensions but got aweights with 2 dimensions
 109. [torch.Tensor.cov]cov(): expected fweights to have one or fewer dimensions but got fweights with 9 dimensions
 110. [torch.Tensor.cov]cov(): expected input to have two or fewer dimensions but got an input with 8 dimensions
 111. [torch.Tensor.cummax]Dimension out of range (expected to be in range of [-2, 1], but got 98)
 112. [torch.Tensor.cummin]Dimension out of range (expected to be in range of [-1, 0], but got 75)
 113. [torch.Tensor.cumprod]Dimension out of range (expected to be in range of [-1, 0], but got 38)
 114. [torch.Tensor.cumsum]Dimension out of range (expected to be in range of [-1, 0], but got 5)
 115. [torch.Tensor.cumsum_]Dimension out of range (expected to be in range of [-4, 3], but got 81)
 116. [torch.Tensor.det]linalg.det: A must be batches of square matrices, but they are 3 by 7 matrices
 117. [torch.Tensor.det]linalg.det: The input tensor A must have at least 2 dimensions.
 118. [torch.Tensor.diag]diag(): Supports 1D or 2D tensors. Got 3D
 119. [torch.Tensor.diag_embed]Dimension out of range (expected to be in range of [-2, 1], but got 87)
 120. [torch.Tensor.diag_embed]diagonal dimensions cannot be identical 0, 0
 121. [torch.Tensor.diagonal]Too large tensor shape: shape = [8, 9, 9, 9, 9, 9, 9, 9, 9]
 122. [torch.Tensor.diagonal]diagonal dimensions cannot be identical -1, 0
 123. [torch.Tensor.diagonal]Dimension out of range (expected to be in range of [-1, 0], but got 51)
 124. [torch.Tensor.diff]diff expects the shape of tensor to prepend or append to match that of input except along the differencing dimension; input.size(1) = 7, but got tensor.size(1) = 5
 125. [torch.Tensor.diff]order must be non-negative but got -4
 126. [torch.Tensor.diff]diff expects prepend or append to be the same dimension as input
 127. [torch.Tensor.diff]Too large tensor shape: shape = [9, 8, 9, 7, 9, 9, 9, 9, 9]
 128. [torch.Tensor.diff]diff expects input to be at least one-dimensional
 129. [torch.Tensor.diff]Dimension out of range (expected to be in range of [-9, 8], but got 96)
 130. [torch.Tensor.dist]Too large tensor shape: shape = [5, 9, 9, 9, 9, 9, 9, 9, 9]
 131. [torch.Tensor.dist]The size of tensor a (8) must match the size of tensor b (5) at non-singleton dimension 8
 132. [torch.Tensor.div]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 5
 133. [torch.Tensor.div_]The size of tensor a (6) must match the size of tensor b (5) at non-singleton dimension 2
 134. [torch.Tensor.div_]Too large tensor shape: shape = [8, 9, 9, 9, 9, 9, 9, 9, 9]
 135. [torch.Tensor.div_]output with shape [8, 9, 9, 9] doesn't match the broadcast shape [7, 8, 9, 9, 9]
 136. [torch.Tensor.divide]The size of tensor a (9) must match the size of tensor b (4) at non-singleton dimension 6
 137. [torch.Tensor.divide_]output with shape [] doesn't match the broadcast shape [5, 6]
 138. [torch.Tensor.dot]dot : expected both vectors to have same dtype, but found Int and Float
 139. [torch.Tensor.dot]inconsistent tensor size, expected tensor [3] and src [9] to have the same number of elements, but got 3 and 9 elements respectively
 140. [torch.Tensor.dot]1D tensors expected, but got 5D and 7D tensors
 141. [torch.Tensor.dsplit]number of sections must be larger than 0, got -1
 142. [torch.Tensor.dsplit]torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with 2 dimensions!
 143. [torch.Tensor.eq]Too large tensor shape: shape = [7, 9, 6, 8, 9, 9, 9, 9, 9]
 144. [torch.Tensor.expand]numel: integer multiplication overflow
 145. [torch.Tensor.expand]expand(torch.FloatTensor{[1, 1]}, size=[]): the number of sizes provided (0) must be greater or equal to the number of dimensions in the tensor (2)
 146. [torch.Tensor.expand_as]The expanded size of the tensor (8) must match the existing size (5) at non-singleton dimension 5.  Target sizes: [7, 8, 8, 9, 8, 8].  Tensor sizes: [5, 5, 5, 5]
 147. [torch.Tensor.flatten]Too large tensor shape: shape = [9, 9, 9, 9, 9, 9, 9, 9, 9]
 148. [torch.Tensor.flatten]flatten() has invalid args: start_dim cannot come after end_dim
 149. [torch.Tensor.flatten]Dimension out of range (expected to be in range of [-4, 3], but got 9)
 150. [torch.Tensor.flip]negative dimensions are not allowed
 151. [torch.Tensor.flip]Too large tensor shape: shape = [7, 8, 9, 9, 9, 9, 9, 9, 9]
 152. [torch.Tensor.flip]dim 8 appears multiple times in the list of dims
 153. [torch.Tensor.float_power]Too large tensor shape: shape = [8, 9, 9, 9, 9, 9, 9, 9, 9]
 154. [torch.Tensor.float_power_]output with shape [] doesn't match the broadcast shape [4, 8, 8, 9, 8]
 155. [torch.Tensor.float_power_]the base given to float_power_ has dtype Float but the operation's result requires dtype Double
 156. [torch.Tensor.float_power_]Too large tensor shape: shape = [9, 8, 6, 7, 9, 8, 9, 9, 9]
 157. [torch.Tensor.floor_divide]Too large tensor shape: shape = [4, 9, 9, 9, 9, 9, 9, 9, 9]
 158. [torch.Tensor.floor_divide]The size of tensor a (7) must match the size of tensor b (2) at non-singleton dimension 5
 159. [torch.Tensor.fmax]Too large tensor shape: shape = [8, 9, 9, 9, 9, 6, 9, 9, 9]
 160. [torch.Tensor.fmin]Too large tensor shape: shape = [9, 9, 9, 9, 9, 9, 9, 9, 9]
 161. [torch.Tensor.fmin]The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 2
 162. [torch.Tensor.fmod]Too large tensor shape: shape = [4, 9, 9, 9, 9, 9, 9, 9, 9]
 163. [torch.Tensor.fmod]The size of tensor a (2) must match the size of tensor b (8) at non-singleton dimension 3
 164. [torch.Tensor.fmod_]result type Float can't be cast to the desired output type Int
 165. [torch.Tensor.fmod_]output with shape [7, 7] doesn't match the broadcast shape [1, 7, 7]
 166. [torch.Tensor.gather]gather(): Expected dtype int64 for index
 167. [torch.Tensor.gather]Dimension out of range (expected to be in range of [-2, 1], but got 78)
 168. [torch.Tensor.gather]Too large tensor shape: shape = [5, 9, 9, 9, 9, 9, 9, 9, 9]
 169. [torch.Tensor.gather]Index tensor must have the same number of dimensions as input tensor
 170. [torch.Tensor.gather]Size does not match at dimension 1 expected index [4, 4] to be smaller than self [1, 1] apart from dimension 0
 171. [torch.Tensor.gather]negative dimensions are not allowed
 172. [torch.Tensor.gather]index 789643 is out of bounds for dimension 0 with size 6
 173. [torch.Tensor.ge]The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 6
 174. [torch.Tensor.gt]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 3
 175. [torch.Tensor.hardshrink]hardshrink_cpu not implemented for 'Long'
 176. [torch.Tensor.hardshrink]hardshrink_cpu not implemented for 'Int'
 177. [torch.Tensor.heaviside]heaviside is not yet implemented for tensors with different dtypes.
 178. [torch.Tensor.heaviside]The size of tensor a (2) must match the size of tensor b (7) at non-singleton dimension 6
 179. [torch.Tensor.heaviside_]heaviside is not yet implemented for tensors with different dtypes.
 180. [torch.Tensor.heaviside_]output with shape [8, 2, 9, 8, 5, 8] doesn't match the broadcast shape [1, 8, 2, 9, 8, 5, 8]
 181. [torch.Tensor.index_select]Dimension out of range (expected to be in range of [-5, 4], but got 24)
 182. [torch.Tensor.index_select]index_select(): Index is supposed to be a vector
 183. [torch.Tensor.index_select]index_select(): Expected dtype int32 or int64 for index
 184. [torch.Tensor.index_select]index out of range in self
 185. [torch.Tensor.ldexp]The size of tensor a (9) must match the size of tensor b (7) at non-singleton dimension 2
 186. [torch.Tensor.ldexp_]result type Float can't be cast to the desired output type Int
 187. [torch.Tensor.ldexp_]output with shape [] doesn't match the broadcast shape [2, 2, 2, 2, 9, 2, 2]
 188. [torch.Tensor.ldexp_]The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 2
 189. [torch.Tensor.le]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 3
 190. [torch.Tensor.lerp]The size of tensor a (9) must match the size of tensor b (7) at non-singleton dimension 6
 191. [torch.Tensor.lerp]lerp_kernel_scalar not implemented for 'Int'
 192. [torch.Tensor.lerp]expected dtype float for `end` but got dtype int
 193. [torch.Tensor.log_softmax]softmax_kernel_impl not implemented for 'Int'
 194. [torch.Tensor.log_softmax]Dimension out of range (expected to be in range of [-2, 1], but got 38)
 195. [torch.Tensor.logical_and]The size of tensor a (2) must match the size of tensor b (6) at non-singleton dimension 4
 196. [torch.Tensor.logical_or]The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 6
 197. [torch.Tensor.logical_or_]The size of tensor a (5) must match the size of tensor b (3) at non-singleton dimension 5
 198. [torch.Tensor.logical_or_]output with shape [] doesn't match the broadcast shape [9, 9, 9, 5, 9, 9, 9]
 199. [torch.Tensor.logical_xor]The size of tensor a (2) must match the size of tensor b (6) at non-singleton dimension 3
 200. [torch.Tensor.logical_xor_]The size of tensor a (7) must match the size of tensor b (4) at non-singleton dimension 6
 201. [torch.Tensor.logical_xor_]output with shape [6, 6] doesn't match the broadcast shape [1, 6, 6]
 202. [torch.Tensor.lt]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 5
 203. [torch.Tensor.masked_fill]masked_fill_ only supports boolean masks, but got mask with dtype float
 204. [torch.Tensor.masked_fill]Too large tensor shape: shape = [8, 9, 9, 9, 9, 9, 9, 9, 9]
 205. [torch.Tensor.masked_fill_]Too large tensor shape: shape = [6, 9, 9, 9, 9, 9, 9, 9, 9]
 206. [torch.Tensor.masked_fill_]masked_fill_ only supports boolean masks, but got mask with dtype float
 207. [torch.Tensor.masked_scatter]masked_scatter: expected self and source to have same dtypes but gotInt and Float
 208. [torch.Tensor.masked_scatter]masked_scatter_ only supports boolean masks, but got mask with dtype Float
 209. [torch.Tensor.masked_scatter]Too large tensor shape: shape = [9, 9, 9, 9, 9, 9, 9, 9, 9]
 210. [torch.Tensor.masked_select]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 8
 211. [torch.Tensor.masked_select]masked_select: expected BoolTensor for mask
 212. [torch.Tensor.masked_select]Too large tensor shape: shape = [8, 9, 7, 8, 9, 9, 7, 9, 9]
 213. [torch.Tensor.matmul]expected scalar type Float but found Int
 214. [torch.Tensor.matmul]mat1 and mat2 shapes cannot be multiplied (30240x9 and 1x6)
 215. [torch.Tensor.matmul]The size of tensor a (6) must match the size of tensor b (7) at non-singleton dimension 6
 216. [torch.Tensor.matmul]both arguments to matmul need to be at least 1D, but they are 2D and 0D
 217. [torch.Tensor.matmul]Too large tensor shape: shape = [8, 8, 8, 8, 8, 8, 9, 8, 8]
 218. [torch.Tensor.matmul]size mismatch, got input (56), mat (56x8), vec (2)
 219. [torch.Tensor.max]The size of tensor a (9) must match the size of tensor b (4) at non-singleton dimension 5
 220. [torch.Tensor.maximum]Too large tensor shape: shape = [4, 9, 9, 9, 9, 9, 9, 9, 9]
 221. [torch.Tensor.maximum]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 4
 222. [torch.Tensor.min]Too large tensor shape: shape = [5, 9, 9, 9, 9, 9, 9, 9, 9]
 223. [torch.Tensor.minimum]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 3
 224. [torch.Tensor.mm]expected m1 and m2 to have the same dtype, but got: int != float
 225. [torch.Tensor.mm]mat1 and mat2 shapes cannot be multiplied (8x7 and 9x9)
 226. [torch.Tensor.mm]mat2 must be a matrix
 227. [torch.Tensor.mm]self must be a matrix
 228. [torch.Tensor.mode]Too large tensor shape: shape = [9, 8, 8, 8, 8, 8, 8, 8, 8]
 229. [torch.Tensor.mode]mode(): Expected reduction dim 0 to have non-zero size.
 230. [torch.Tensor.mode]Dimension out of range (expected to be in range of [-1, 0], but got 5)
 231. [torch.Tensor.moveaxis]negative dimensions are not allowed
 232. [torch.Tensor.moveaxis]movedim: repeated dim in `source` ([-1, -1, -1, -1, -3])
 233. [torch.Tensor.moveaxis]Too large tensor shape: shape = [4, 8, 9, 9, 9, 9, 9, 9, 9]
 234. [torch.Tensor.moveaxis]Dimension out of range (expected to be in range of [-8, 7], but got 9)
 235. [torch.Tensor.moveaxis]movedim: Invalid source or destination dims: source ([-2, -4, -3, -3, -3, -3, -3] dims) should contain the same number of dims as destination ([-3, 9, -1, 9, 3, 4, 9, 9] dims)
 236. [torch.Tensor.movedim]Dimension out of range (expected to be in range of [-2, 1], but got 37)
 237. [torch.Tensor.mul]The size of tensor a (2) must match the size of tensor b (6) at non-singleton dimension 3
 238. [torch.Tensor.mul_]The size of tensor a (9) must match the size of tensor b (2) at non-singleton dimension 6
 239. [torch.Tensor.mul_]output with shape [] doesn't match the broadcast shape [3, 1, 2, 1]
 240. [torch.Tensor.mul_]Too large tensor shape: shape = [8, 9, 8, 8, 8, 8, 8, 8, 8]
 241. [torch.Tensor.mul_]result type Float can't be cast to the desired output type Int
 242. [torch.Tensor.mv]expected scalar type Int but found Float
 243. [torch.Tensor.mv]size mismatch, got input (3), mat (3x6), vec (8)
 244. [torch.Tensor.mv]Too large tensor shape: shape = [8, 8, 8, 8, 8, 8, 8, 9, 8]
 245. [torch.Tensor.mv]Dimension specified as 0 but tensor has no dimensions
 246. [torch.Tensor.mvlgamma]p has to be greater than or equal to 1
 247. [torch.Tensor.nanmean]nanmean(): expected input to have floating point or complex dtype but got Int
 248. [torch.Tensor.narrow]Dimension out of range (expected to be in range of [-2, 1], but got 4)
 249. [torch.Tensor.narrow]narrow() cannot be applied to a 0-dim tensor.
 250. [torch.Tensor.narrow]negative dimensions are not allowed
 251. [torch.Tensor.narrow]start (7) + length (8) exceeds dimension size (9).
 252. [torch.Tensor.ne]The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 2
 253. [torch.Tensor.new_full]Need to provide pin_memory allocator to use pin memory.
 254. [torch.Tensor.new_ones]Need to provide pin_memory allocator to use pin memory.
 255. [torch.Tensor.new_zeros]Need to provide pin_memory allocator to use pin memory.
 256. [torch.Tensor.norm]linalg.vector_norm cannot compute the -3 norm on an empty tensor because the operation does not have an identity
 257. [torch.Tensor.norm]linalg.vector_norm: Expected a floating point or complex tensor as input. Got Int
 258. [torch.Tensor.permute]permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 2 is not equal to len(dims) = 0
 259. [torch.Tensor.permute]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 260. [torch.Tensor.polygamma_]result type Float can't be cast to the desired output type Int
 261. [torch.Tensor.polygamma_]polygamma(n, x) does not support negative n.
 262. [torch.Tensor.pow]The size of tensor a (9) must match the size of tensor b (7) at non-singleton dimension 6
 263. [torch.Tensor.pow_]Integers to negative integer powers are not allowed.
 264. [torch.Tensor.reciprocal_]result type Float can't be cast to the desired output type Int
 265. [torch.Tensor.remainder_]The size of tensor a (6) must match the size of tensor b (5) at non-singleton dimension 2
 266. [torch.Tensor.remainder_]output with shape [] doesn't match the broadcast shape [2, 4, 4, 4, 4, 4, 4]
 267. [torch.Tensor.remainder_]result type Float can't be cast to the desired output type Int
 268. [torch.Tensor.repeat]Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor
 269. [torch.Tensor.repeat]Trying to create tensor with negative dimension -4: [-4, -4, -4, -4, 5, -4, -4, -4, -4]
 270. [torch.Tensor.reshape_as]shape '[1, 1, 7, 6, 6, 2, 1]' is invalid for input of size 1
 271. [torch.Tensor.rot90]expected total rotation dims == 2, but got dims = 0
 272. [torch.Tensor.rsqrt_]result type Float can't be cast to the desired output type Int
 273. [torch.Tensor.scatter_]Dimension out of range (expected to be in range of [-2, 1], but got 9)
 274. [torch.Tensor.scatter_]scatter(): Expected dtype int64 for index
 275. [torch.Tensor.scatter_]Index tensor must have the same number of dimensions as self tensor
 276. [torch.Tensor.scatter_]Expected index [8, 5, 6, 2, 4] to be smaller than self [3, 2, 8, 8, 4] apart from dimension 1 and to be smaller size than src [6, 1, 6, 6, 9]
 277. [torch.Tensor.scatter_add_]Dimension out of range (expected to be in range of [-5, 4], but got 33)
 278. [torch.Tensor.scatter_add_]scatter(): Expected dtype int64 for index
 279. [torch.Tensor.scatter_add_]Index tensor must have the same number of dimensions as self tensor
 280. [torch.Tensor.scatter_add_]Expected index [7, 5, 5, 5, 9, 5, 5] to be smaller than self [9, 8, 8, 8, 8, 8, 8] apart from dimension 3 and to be smaller size than src [7, 6, 6, 6, 9, 6, 6]
 281. [torch.Tensor.scatter_add_]index -406719 is out of bounds for dimension 0 with size 2
 282. [torch.Tensor.scatter_reduce_]Dimension out of range (expected to be in range of [-4, 3], but got 5)
 283. [torch.Tensor.scatter_reduce_]scatter(): Expected dtype int64 for index
 284. [torch.Tensor.scatter_reduce_]Index tensor must have the same number of dimensions as self tensor
 285. [torch.Tensor.scatter_reduce_]Expected index [3, 6, 6, 6, 6] to be smaller than self [1, 9, 9, 9, 9] apart from dimension 1 and to be smaller size than src [7, 4, 4, 4, 4]
 286. [torch.Tensor.scatter_reduce_]reduce argument must be either sum, prod, mean, amax or amin, got gEbJ
 287. [torch.Tensor.sigmoid_]result type Float can't be cast to the desired output type Int
 288. [torch.Tensor.sin_]result type Float can't be cast to the desired output type Int
 289. [torch.Tensor.sinh_]result type Float can't be cast to the desired output type Int
 290. [torch.Tensor.softmax]Dimension out of range (expected to be in range of [-5, 4], but got 6)
 291. [torch.Tensor.sort]Dimension out of range (expected to be in range of [-7, 6], but got 9)
 292. [torch.Tensor.sqrt_]result type Float can't be cast to the desired output type Int
 293. [torch.Tensor.sub]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 6
 294. [torch.Tensor.sub_]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 6
 295. [torch.Tensor.swapaxes_]Dimension out of range (expected to be in range of [-5, 4], but got 24)
 296. [torch.Tensor.swapdims_]Dimension out of range (expected to be in range of [-7, 6], but got 8)
 297. [torch.Tensor.t]t() expects a tensor with <= 2 dimensions, but self is 5D
 298. [torch.Tensor.t_]t_() expects a tensor with <= 2 dimensions, but self is 3D
 299. [torch.Tensor.tan_]result type Float can't be cast to the desired output type Int
 300. [torch.Tensor.tanh_]result type Float can't be cast to the desired output type Int
 301. [torch.Tensor.transpose]Dimension out of range (expected to be in range of [-2, 1], but got 71)
 302. [torch.Tensor.transpose_]negative dimensions are not allowed
 303. [torch.Tensor.transpose_]Dimension out of range (expected to be in range of [-2, 1], but got 24)
 304. [torch.Tensor.tril]tril: input tensor must have at least 2 dimensions
 305. [torch.Tensor.tril_]tril: input tensor must have at least 2 dimensions
 306. [torch.Tensor.triu_]triu: input tensor must have at least 2 dimensions
 307. [torch.Tensor.true_divide_]result type Float can't be cast to the desired output type Int
 308. [torch.Tensor.true_divide_]The size of tensor a (4) must match the size of tensor b (7) at non-singleton dimension 6
 309. [torch.Tensor.unfold]step is 0 but must be > 0
 310. [torch.Tensor.unfold]negative dimensions are not allowed
 311. [torch.Tensor.unfold]Storage size calculation overflowed with sizes=[1, 9, 9, 9, 9, 14, -4] and strides=[59049, 6561, 729, 81, 9, 1, 1]
 312. [torch.Tensor.unfold]maximum size for tensor at dimension 1 is 4 but size is 8
 313. [torch.Tensor.unfold]Dimension out of range (expected to be in range of [-2, 1], but got 33)
 314. [torch.Tensor.unique_consecutive]Dimension specified as 0 but tensor has no dimensions
 315. [torch.Tensor.unique_consecutive]Dimension out of range (expected to be in range of [-1, 0], but got 57)
 316. [torch.Tensor.unsqueeze]Dimension out of range (expected to be in range of [-3, 2], but got 91)
 317. [torch.Tensor.unsqueeze_]Dimension out of range (expected to be in range of [-3, 2], but got 74)
 318. [torch.Tensor.view]invalid shape dimension -4
 319. [torch.Tensor.view_as]shape '[7, 9, 8, 7, 7]' is invalid for input of size 6561
 320. [torch.Tensor.xlogy_]result type Float can't be cast to the desired output type Int
 321. [torch.Tensor.xlogy_]output with shape [5, 5] doesn't match the broadcast shape [1, 5, 5]
 322. [torch.Tensor.xlogy_]The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 6
 323. [torch._C._fft.fft_fft]MKL FFT error: Intel MKL DFTI ERROR: Inconsistent configuration parameters
 324. [torch._C._fft.fft_fft]negative dimensions are not allowed
 325. [torch._C._fft.fft_fft]Invalid normalization mode: jfhs
 326. [torch._C._fft.fft_fft]Invalid number of data points (-2) specified
 327. [torch._C._fft.fft_fft]Dimension specified as 1 but tensor has no dimensions
 328. [torch._C._fft.fft_fft]Dimension out of range (expected to be in range of [-1, 0], but got 1)
 329. [torch._C._fft.fft_fft]fft expects a complex output tensor, but got Float
 330. [torch._C._fft.fft_fft]Trying to resize storage that is not resizable
 331. [torch._C._fft.fft_fft2]Invalid normalization mode: wGzo
 332. [torch._C._fft.fft_fft2]Dimension out of range (expected to be in range of [-2, 1], but got 29)
 333. [torch._C._fft.fft_fft2]fftn expects a complex output tensor, but got Float
 33
 335. [torch._C._fft.fft_fftn]Invalid normalization mode: cipM
 336. [torch._C._fft.fft_fftn]Trying to resize storage that is not resizable
 337. [torch._C._fft.fft_fftn]fftn expects a complex output tensor, but got Float
 338. [torch._C._fft.fft_hfft]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 339. [torch._C._fft.fft_hfft]Invalid number of data points (0) specified
 340. [torch._C._fft.fft_hfft]Trying to resize storage that is not resizable
 341. [torch._C._fft.fft_hfft]Dimension specified as 0 but tensor has no dimensions
 342. [torch._C._fft.fft_hfft]Invalid normalization mode: zxLX
 343. [torch._C._fft.fft_hfft2]hfftn must transform at least one axis
 344. [torch._C._fft.fft_hfft2]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 345. [torch._C._fft.fft_hfft2]Dimension specified as 3 but tensor has no dimensions
 346. [torch._C._fft.fft_hfftn]hfftn must transform at least one axis
 347. [torch._C._fft.fft_hfftn]Invalid normalization mode: UbxY
 348. [torch._C._fft.fft_hfftn]Trying to resize storage that is not resizable
 349. [torch._C._fft.fft_ifft]Dimension out of range (expected to be in range of [-2, 1], but got -3)
 350. [torch._C._fft.fft_ifft]Invalid normalization mode: TNkG
 351. [torch._C._fft.fft_ifft]Invalid number of data points (-3) specified
 352. [torch._C._fft.fft_ifft]ifft expects a complex output tensor, but got Float
 353. [torch._C._fft.fft_ifft]Found dtype ComplexDouble but expected ComplexFloat
 354. [torch._C._fft.fft_ifft]Trying to resize storage that is not resizable
 355. [torch._C._fft.fft_ifft]Dimension specified as 9 but tensor has no dimensions
 356. [torch._C._fft.fft_ifft2]Invalid normalization mode: Gyck
 357. [torch._C._fft.fft_ifft2]Dimension out of range (expected to be in range of [-1, 0], but got 3)
 358. [torch._C._fft.fft_ifft2]Dimension specified as -3 but tensor has no dimensions
 359. [torch._C._fft.fft_ifft2]Trying to resize storage that is not resizable
 360. [torch._C._fft.fft_ifft2]fftn expects a complex output tensor, but got Float
 361. [torch._C._fft.fft_ifftn]Invalid normalization mode: CMPv
 362. [torch._C._fft.fft_ifftn]fftn expects a complex output tensor, but got Float
 36
 364. [torch._C._fft.fft_ihfft]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 365. [torch._C._fft.fft_ihfft]Invalid normalization mode: kElB
 366. [torch._C._fft.fft_ihfft]ihfft expects a complex output tensor, but got Float
 367. [torch._C._fft.fft_ihfft2]ihfftn must transform at least one axis
 368. [torch._C._fft.fft_ihfft2]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 369. [torch._C._fft.fft_ihfftn]ihfftn must transform at least one axis
 370. [torch._C._fft.fft_ihfftn]Invalid normalization mode: u3WjXW76AK
 371. [torch._C._fft.fft_ihfftn]ihfftn expects a complex output tensor, but got Float
 372. [torch._C._fft.fft_irfft]Dimension out of range (expected to be in range of [-2, 1], but got -4)
 379. [torch._C._fft.fft_irfft2]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 380. [torch._C._fft.fft_irfftn]Invalid normalization mode: yYGc
 381. [torch._C._fft.fft_irfftn]Trying to resize storage that is not resizable
 382. [torch._C._fft.fft_irfftn]irfftn must transform at least one axis
 383. [torch._C._fft.fft_rfft]Dimension out of range (expected to be in range of [-2, 1], but got 2)
 384. [torch._C._fft.fft_rfft]Invalid normalization mode: lDny
 385. [torch._C._fft.fft_rfft]rfft expects a complex output tensor, but got Float
 386. [torch._C._fft.fft_rfft]Trying to resize storage that is not resizable
 387. [torch._C._fft.fft_rfft]Dimension specified as 0 but tensor has no dimensions
 388. [torch._C._fft.fft_rfft2]rfftn must transform at least one axis
 389. [torch._C._fft.fft_rfft2]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 390. [torch._C._fft.fft_rfft2]Invalid normalization mode: mqLo
 391. [torch._C._fft.fft_rfft2]rfftn expects a complex output tensor, but got Float
 392. [torch._C._fft.fft_rfftn]Invalid normalization mode: CwJA
 393. [torch._C._fft.fft_rfftn]rfftn must transform at least one axis
 394. [torch._C._fft.fft_rfftn]rfftn expects a complex output tensor, but got Float
 395. [torch._C._fft.fft_rfftn]Trying to resize storage that is not resizable
 396. [torch._C._linalg.linalg_cholesky]linalg.cholesky: A must be batches of square matrices, but they are 3 by 2 matrices
 397. [torch._C._linalg.linalg_cholesky]linalg.cholesky: The input tensor A must have at least 2 dimensions.
 398. [torch._C._linalg.linalg_cholesky]Trying to resize storage that is not resizable
 399. [torch._C._linalg.linalg_cond]linalg.svd: Expected a floating point or complex tensor as input. Got Int
 400. [torch._C._linalg.linalg_cond]linalg.cond: The input tensor must have at least 2 dimensions.
 401. [torch._C._linalg.linalg_cond]linalg.cond got an invalid norm type: 4
 402. [torch._C._linalg.linalg_cond]linalg.cond(ord=fro): A must be batches of square matrices, but they are 2 by 9 matrices
 403. [torch._C._linalg.linalg_cond]linalg.inv: Low precision dtypes not supported. Got Half
 404. [torch._C._linalg.linalg_cond]linalg.cond: Expected result to be safely castable from Float dtype, but got result with dtype Int
 405. [torch._C._linalg.linalg_cond]Trying to resize storage that is not resizable
 406. [torch._C._linalg.linalg_det]linalg.det: Expected a floating point or complex tensor as input. Got Int
 407. [torch._C._linalg.linalg_det]linalg.det: A must be batches of square matrices, but they are 5 by 7 matrices
 408. [torch._C._linalg.linalg_det]linalg.det: The input tensor A must have at least 2 dimensions.
 409. [torch._C._linalg.linalg_det]Expected out tensor to have dtype c10::complex<float>, but got float instead
 410. [torch._C._linalg.linalg_det]Trying to resize storage that is not resizable
 411. [torch._C._linalg.linalg_eigh]Expected UPLO argument to be 'L' or 'U', but got Ypgg
 412. [torch._C._linalg.linalg_eigvals]linalg.eigvals: A must be batches of square matrices, but they are 1 by 4 matrices
 413. [torch._C._linalg.linalg_eigvals]linalg.eigvals: The input tensor A must have at least 2 dimensions.
 414. [torch._C._linalg.linalg_eigvals]torch.linalg.eigvals: Expected eigenvalues to be safely castable from ComplexFloat dtype, but got eigenvalues with dtype Float
 41
 416. [torch._C._linalg.linalg_eigvalsh]Expected UPLO argument to be 'L' or 'U', but got pGNG
 417. [torch._C._linalg.linalg_eigvalsh]Expected out tensor to have dtype float, but got c10::Half instead
 418. [torch._C._linalg.linalg_eigvalsh]negative dimensions are not allowed
 419. [torch._C._linalg.linalg_eigvalsh]Trying to resize storage that is not resizable
 420. [torch._C._linalg.linalg_eigvalsh]linalg.eigh: The input tensor A must have at least 2 dimensions.
 421. [torch._C._linalg.linalg_eigvalsh]linalg.eigh: A must be batches of square matrices, but they are 1 by 6 matrices
 422. [torch._C._linalg.linalg_inv]linalg.inv: Expected a floating point or complex tensor as input. Got Int
 423. [torch._C._linalg.linalg_inv]linalg.inv: A must be batches of square matrices, but they are 5 by 8 matrices
 424. [torch._C._linalg.linalg_inv]linalg.inv: The input tensor A must have at least 2 dimensions.
 425. [torch._C._linalg.linalg_inv_ex]linalg.inv: A must be batches of square matrices, but they are 6 by 2 matrices
 426. [torch._C._linalg.linalg_inv_ex]linalg.inv: The input tensor A must have at least 2 dimensions.
 427. [torch._C._linalg.linalg_ldl_factor]torch.linalg.ldl_factor_ex: A must be batches of square matrices, but they are 9 by 3 matrices
 428. [torch._C._linalg.linalg_ldl_factor]torch.linalg.ldl_factor_ex: The input tensor A must have at least 2 dimensions.
 429. [torch._C._linalg.linalg_ldl_factor_ex]torch.linalg.ldl_factor_ex: A must be batches of square matrices, but they are 7 by 8 matrices
 430. [torch._C._linalg.linalg_ldl_factor_ex]torch.linalg.ldl_factor_ex: The input tensor A must have at least 2 dimensions.
 431. [torch._C._linalg.linalg_lstsq]torch.linalg.lstsq: input.dim() must be greater or equal to other.dim() and (input.dim() - other.dim()) <= 1
 432. [torch._C._linalg.linalg_lstsq]torch.linalg.lstsq: input must have at least 2 dimensions.
 433. [torch._C._linalg.linalg_lstsq]torch.linalg.lstsq: input.size(-2) should match other.size(-2)
 434. [torch._C._linalg.linalg_lstsq]torch.linalg.lstsq: parameter `driver` should be one of (gels, gelsy, gelsd, gelss)
 435. [torch._C._linalg.linalg_lstsq]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 1
 436. [torch._C._linalg.linalg_lstsq]torch.linalg.lstsq: Expected input and other to have the same dtype, but got input's dtype Int and other's dtype Float
 437. [torch._C._linalg.linalg_lu]lu_cpu not implemented for 'Int'
 438. [torch._C._linalg.linalg_lu]linalg.lu: Expected tensor with 2 or more dimensions. Got size: [] instead
 439. [torch._C._linalg.linalg_lu]linalg.lu_factor: LU without pivoting is not implemented on the CPU
 440. [torch._C._linalg.linalg_lu_factor]lu_cpu not implemented for 'Int'
 441. [torch._C._linalg.linalg_lu_factor]torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: [] instead
 442. [torch._C._linalg.linalg_lu_factor]linalg.lu_factor: LU without pivoting is not implemented on the CPU
 44
 444. [torch._C._linalg.linalg_lu_factor_ex]lu_cpu not implemented for 'Bool'
 445. [torch._C._linalg.linalg_lu_factor_ex]lu_cpu not implemented for 'Long'
 446. [torch._C._linalg.linalg_lu_factor_ex]lu_cpu not implemented for 'Int'
 447. [torch._C._linalg.linalg_lu_factor_ex]torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: [] instead
 448. [torch._C._linalg.linalg_lu_factor_ex]linalg.lu_factor: LU without pivoting is not implemented on the CPU
 449. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm: Order 0 not supported.
 450. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm: dim must be a 2-tuple. Got 
 451. [torch._C._linalg.linalg_matrix_norm]Dimension out of range (expected to be in range of [-3, 2], but got 6)
 452. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm: dims must be different. Got (3, 3)
 453. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm expected out tensor dtype Double but got: Float
 454. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm: The input tensor A must have at least 2 dimensions.
 455. [torch._C._linalg.linalg_matrix_norm]Trying to resize storage that is not resizable
 456. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm: Expected a floating point or complex tensor as input. Got Int
 457. [torch._C._linalg.linalg_matrix_power]linalg.matrix_power: The input tensor A must have at least 2 dimensions.
 458. [torch._C._linalg.linalg_matrix_power]linalg.inv: Expected a floating point or complex tensor as input. Got Int
 459. [torch._C._linalg.linalg_matrix_power]Expected out tensor to have dtype int, but got float instead
 460. [torch._C._linalg.linalg_matrix_power]Trying to resize storage that is not resizable
 461. [torch._C._linalg.linalg_matrix_power]linalg.matrix_power: A must be batches of square matrices, but they are 9 by 4 matrices
 462. [torch._C._linalg.linalg_matrix_rank]linalg_eigh_cpu not implemented for 'Int'
 463. [torch._C._linalg.linalg_matrix_rank]linalg.svd: Expected a floating point or complex tensor as input. Got Int
 464. [torch._C._linalg.linalg_matrix_rank]torch.linalg.matrix_rank: The input tensor input must have at least 2 dimensions.
 465. [torch._C._linalg.linalg_matrix_rank]The size of tensor a (5) must match the size of tensor b (8) at non-singleton dimension 6
 466. [torch._C._linalg.linalg_matrix_rank]linalg.eigh: A must be batches of square matrices, but they are 4 by 3 matrices
 467. [torch._C._linalg.linalg_matrix_rank]This function doesn't handle types other than float and double
 468. [torch._C._linalg.linalg_matrix_rank]negative dimensions are not allowed
 469. [torch._C._linalg.linalg_matrix_rank]Trying to resize storage that is not resizable
 470. [torch._C._linalg.linalg_norm]linalg.matrix_norm: Order 18 not supported.
 471. [torch._C._linalg.linalg_norm]linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got 4D.
 472. [torch._C._linalg.linalg_norm]linalg.matrix_norm: Expected a floating point or complex tensor as input. Got Int
 473. [torch._C._linalg.linalg_norm]linalg.matrix_norm: The input tensor A must have at least 2 dimensions.
 474. [torch._C._linalg.linalg_norm]linalg.norm expected out tensor dtype Double but got: Float
 475. [torch._C._linalg.linalg_norm]Trying to resize storage that is not resizable
 476. [torch._C._linalg.linalg_pinv]linalg.pinv(Int{[3, 5, 5, 5, 5, 5, 5]}): expected a tensor with 2 or more dimensions of float, double, cfloat or cdouble types
 477. [torch._C._linalg.linalg_pinv]linalg.eigh: A must be batches of square matrices, but they are 1 by 3 matrices
 478. [torch._C._linalg.linalg_pinv]The size of tensor a (3) must match the size of tensor b (9) at non-singleton dimension 6
 479. [torch._C._linalg.linalg_pinv]Trying to resize storage that is not resizable
 480. [torch._C._linalg.linalg_pinv]linalg.pinv: Expected result to be safely castable from ComplexFloat dtype, but got result with dtype Float
 481. [torch._C._linalg.linalg_pinv]negative dimensions are not allowed
 482. [torch._C._linalg.linalg_qr]linalg.qr: Expected a floating point or complex tensor as input. Got Int
 483. [torch._C._linalg.linalg_qr]linalg.qr: The input tensor A must have at least 2 dimensions.
 484. [torch._C._linalg.linalg_qr]qr received unrecognized mode 'RXoC' but expected one of 'reduced' (default), 'r', or 'complete'
 485. [torch._C._linalg.linalg_slogdet]linalg.slogdet: Expected a floating point or complex tensor as input. Got Int
 486. [torch._C._linalg.linalg_slogdet]linalg.slogdet: A must be batches of square matrices, but they are 3 by 2 matrices
 487. [torch._C._linalg.linalg_slogdet]linalg.slogdet: The input tensor A must have at least 2 dimensions.
 488. [torch._C._linalg.linalg_solve]linalg.solve: A must be batches of square matrices, but they are 4 by 6 matrices
 489. [torch._C._linalg.linalg_solve]linalg.solve: The input tensor A must have at least 2 dimensions.
 490. [torch._C._linalg.linalg_solve]linalg.solve: Incompatible shapes of A and B for the equation XA = B (5x5 and 8x8)
 491. [torch._C._linalg.linalg_solve]linalg.solve: Expected a floating point or complex tensor as input. Got Int
 492. [torch._C._linalg.linalg_solve]negative dimensions are not allowed
 493. [torch._C._linalg.linalg_solve]linalg.solve: Expected A and B to have the same dtype, but found A of type Float and B of type Int instead
 494. [torch._C._linalg.linalg_solve]Trying to resize storage that is not resizable
 495. [torch._C._linalg.linalg_solve]The size of tensor a (9) must match the size of tensor b (4) at non-singleton dimension 0
 496. [torch._C._linalg.linalg_solve_ex]linalg.solve: A must be batches of square matrices, but they are 1 by 7 matrices
 497. [torch._C._linalg.linalg_solve_ex]linalg.solve: The input tensor A must have at least 2 dimensions.
 498. [torch._C._linalg.linalg_solve_ex]linalg.solve: Incompatible shapes of A and B for the equation AX = B (5x5 and 6x6)
 499. [torch._C._linalg.linalg_solve_triangular]linalg.solve_triangular: Incompatible shapes of A and B for the equation XA = B (8x8 and 1x1)
 500. [torch._C._linalg.linalg_solve_triangular]linalg.solve_triangular: The input tensor B must have at least 2 dimensions.
 501. [torch._C._linalg.linalg_solve_triangular]linalg.solve_triangular: A must be batches of square matrices, but they are 6 by 7 matrices
 502. [torch._C._linalg.linalg_solve_triangular]negative dimensions are not allowed
 503. [torch._C._linalg.linalg_solve_triangular]triangular_solve_cpu not implemented for 'Half'
 504. [torch._C._linalg.linalg_solve_triangular]triangular_solve_cpu not implemented for 'Int'
 505. [torch._C._linalg.linalg_svd]torch.linalg.svd: keyword argument `driver=` is only supported on CUDA inputs with cuSOLVER backend.
 506. [torch._C._linalg.linalg_svdvals]torch.linalg.svd: keyword argument `driver=` is only supported on CUDA inputs with cuSOLVER backend.
 507. [torch._C._linalg.linalg_tensorinv]ArrayRef: invalid slice, N = 7; size = 2
 508. [torch._C._linalg.linalg_tensorinv]Expected a strictly positive integer for 'ind', but got 0
 509. [torch._C._linalg.linalg_tensorsolve]cannot create std::vector larger than max_size()
 510. [torch._C._linalg.linalg_vander]N must be greater than 1.
 511. [torch._C._linalg.linalg_vecdot]linalg.vecdot: Expected a floating point or complex tensor as input. Got Int
 512. [torch._C._linalg.linalg_vecdot]linalg.vecdot: Expected x and y to have the same dtype, but found x of type Float and y of type Double instead
 513. [torch._C._linalg.linalg_vecdot]The size of tensor a (2) must match the size of tensor b (8) at non-singleton dimension 6
 514. [torch._C._linalg.linalg_vecdot]Dimension out of range (expected to be in range of [-7, 6], but got 86)
 515. [torch._C._linalg.linalg_vecdot]linalg.vecdot: Expected out of dtypeFloat but found Int
 516. [torch._C._linalg.linalg_vector_norm]linalg.vector_norm cannot compute the -3 norm on an empty tensor because the operation does not have an identity
 517. [torch._C._linalg.linalg_vector_norm]linalg.vector_norm: Expected a floating point or complex tensor as input. Got Int
 518. [torch._C._linalg.linalg_vector_norm]Expected out tensor to have dtype float, but got int instead
 519. [torch._C._linalg.linalg_vector_norm]Trying to resize storage that is not resizable
 520. [torch._C._nn.adaptive_max_pool2d]adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes [0, 0, 7, 9] with dimension 1 being empty
 52
 522. [torch._C._nn.adaptive_max_pool2d]adaptive_max_pool2d not implemented for 'Bool'
 523. [torch._C._nn.adaptive_max_pool2d]adaptive_max_pool2d not implemented for 'Int'
 524. [torch._C._nn.adaptive_max_pool2d]Trying to create tensor with negative dimension -2: [8, 8, 5, -2]
 525. [torch._C._nn.adaptive_max_pool2d]adaptive_max_pool2d(): internal error: output_size.size() must be 2
 526. [torch._C._nn.adaptive_max_pool2d]adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: [8, 5, 8, 8, 8, 8, 8]
 527. [torch._C._nn.adaptive_max_pool3d]adaptive_max_pool3d(): Expected 4D or 5D tensor, but got: [1, 1]
 528. [torch._C._nn.adaptive_max_pool3d]adaptive_max_pool3d(): internal error: output_size.size() must be 3
 529. [torch._C._nn.adaptive_max_pool3d]adaptive_max_pool3d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes [9, 8, 7, 0, 6] with dimension 3 being empty
 530. [torch._C._nn.adaptive_max_pool3d]adaptive_max_pool3d_cpu not implemented for 'Int'
 531. [torch._C._nn.avg_pool2d]avg_pool2d: kernel_size must either be a single int, or a tuple of two ints
 532. [torch._C._nn.avg_pool2d]pad should be at most half of effective kernel size, but got pad=8, kernel_size=1 and dilation=1
 533. [torch._C._nn.avg_pool2d]stride should not be zero
 534. [torch._C._nn.avg_pool2d]pad must be non-negative, but got pad: -1
 535. [torch._C._nn.avg_pool2d]stride should be greater than zero, but got dH: 8 dW: -1
 536. [torch._C._nn.avg_pool2d]Dimension specified as -3 but tensor has no dimensions
 537. [torch._C._nn.avg_pool3d]avg_pool3d: kernel_size must be a single int, or a tuple of three ints
 538. [torch._C._nn.avg_pool3d]non-empty 4D or 5D (batch mode) tensor expected for input
 539. [torch._C._nn.avg_pool3d]pad must be non-negative, but got pad: -3
 540. [torch._C._nn.avg_pool3d]pad should be at most half of effective kernel size, but got pad=3, kernel_size=5 and dilation=1
 541. [torch._C._nn.avg_pool3d]stride should be greater than zero, but got dT: 5 dH: -1 dW: -1
 542. [torch._C._nn.flatten_dense_tensors]torch.cat(): expected a non-empty list of Tensors
 543. [torch._C._nn.gelu]GeluKernelImpl not implemented for 'Int'
 544. [torch._C._nn.gelu]approximate argument must be either none or tanh.
 545. [torch._C._nn.gelu]Found dtype Int but expected Float
 546. [torch._C._nn.gelu]Trying to resize storage that is not resizable
 547. [torch._C._nn.gelu_]GeluKernelImpl not implemented for 'Int'
 548. [torch._C._nn.gelu_]approximate argument must be either none or tanh.
 549. [torch._C._nn.huber_loss]huber_cpu not implemented for 'Int'
 550. [torch._C._nn.huber_loss]result type Float can't be cast to the desired output type Int
 551. [torch._C._nn.huber_loss]huber_loss does not support non-positive values for delta.
 552. [torch._C._nn.huber_loss]The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 2
 553. [torch._C._nn.huber_loss]Trying to resize storage that is not resizable
 554. [torch._C._nn.huber_loss]huber_cpu not implemented for 'Long'
 555. [torch._C._nn.huber_loss]huber_cpu not implemented for 'Bool'
 55
 557. [torch._C._nn.l1_loss]The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 6
 558. [torch._C._nn.log_sigmoid]log_sigmoid_cpu not implemented for 'Int'
 559. [torch._C._nn.log_sigmoid]Trying to resize storage that is not resizable
 560. [torch._C._nn.max_pool2d_with_indices]max_pool2d: kernel_size must either be a single int, or a tuple of two ints
 561. [torch._C._nn.max_pool2d_with_indices]non-empty 3D or 4D (batch mode) tensor expected for input
 562. [torch._C._nn.mse_loss]The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 4
 563. [torch._C._nn.pad_sequence]Too large tensor shape: shape = [7, 9, 9, 9, 9, 9, 9, 9, 9]
 564. [torch._C._nn.pad_sequence]The size of tensor a (7) must match the size of tensor b (9) at non-singleton dimension 8
 565. [torch._C._nn.pad_sequence]received an empty list of sequences
 566. [torch._C._nn.reflection_pad1d]negative dimensions are not allowed
 567. [torch._C._nn.reflection_pad1d]Expected 2D or 3D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: [7, 0, 0]
 568. [torch._C._nn.reflection_pad1d]Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (8, 8) at dimension 1 of input [7, 8, 8, 8, 8, 8, 8]
 569. [torch._C._nn.reflection_pad1d]padding size is expected to be 2, but got: 0
 570. [torch._C._nn.reflection_pad1d]reflection_pad1d not implemented for 'Half'
 571. [torch._C._nn.reflection_pad1d]Expected out tensor to have dtype float, but got c10::Half instead
 572. [torch._C._nn.reflection_pad1d]Trying to resize storage that is not resizable
 573. [torch._C._nn.reflection_pad1d]Dimension out of range (expected to be in range of [-1, 0], but got 1)
 574. [torch._C._nn.reflection_pad1d]Dimension specified as 0 but tensor has no dimensions
 575. [torch._C._nn.smooth_l1_loss]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 3
 576. [torch._C._nn.soft_margin_loss]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 2
 577. [torch._C._nn.soft_margin_loss]result type Float can't be cast to the desired output type Int
 578. [torch._C._nn.soft_margin_loss]Found dtype Int but expected Float
 579. [torch._C._nn.softplus]softplus_cpu not implemented for 'Int'
 580. [torch._C._nn.softplus]Found dtype Float but expected Int
 581. [torch._C._nn.softplus]Trying to resize storage that is not resizable
 582. [torch._C._nn.softshrink]softshrink_cpu not implemented for 'Int'
 583. [torch._C._nn.softshrink]lambda must be greater or equal to 0, but found to be -1.
 584. [torch._C._nn.softshrink]Trying to resize storage that is not resizable
 585. [torch._C._nn.softshrink]Found dtype Float but expected Int
 586. [torch._C._nn.upsample_bicubic2d]Non-empty 4D data tensor expected but got a tensor with sizes [7, 0, 8, 9]
 587. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'ComplexFloat'
 588. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'Long'
 589. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'Int'
 590. [torch._C._nn.upsample_bicubic2d]Input and output sizes should be greater than 0, but got input (H: 7, W: 8) output (H: -3, W: 3)
 591. [torch._C._nn.upsample_bicubic2d]It is expected output_size equals to 2, but got size 0
 592. [torch._C._nn.upsample_bicubic2d]Expected out tensor to have dtype int, but got float instead
 593. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'Char'
 594. [torch._C._nn.upsample_bilinear2d]Expected static_cast<int64_t>(output_size->size()) == spatial_dimensions to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 595. [torch._C._nn.upsample_bilinear2d]Non-empty 4D data tensor expected but got a tensor with sizes [9, 0, 3, 8]
 596. [torch._C._nn.upsample_bilinear2d]upsample_bilinear2d_channels_last not implemented for 'Int'
 597. [torch._C._nn.upsample_bilinear2d]Input and output sizes should be greater than 0, but got input (H: 7, W: 1) output (H: -3, W: 3)
 598. [torch._C._nn.upsample_bilinear2d]It is expected output_size equals to 2, but got size 0
 599. [torch._C._nn.upsample_linear1d]Non-empty 3D data tensor expected but got a tensor with sizes [8, 0, 1]
 600. [torch._C._nn.upsample_linear1d]compute_indices_weights_linear not implemented for 'Bool'
 601. [torch._C._nn.upsample_linear1d]compute_indices_weights_linear not implemented for 'Int'
 602. [torch._C._nn.upsample_linear1d]Input and output sizes should be greater than 0, but got input (W: 8) and output (W: -3)
 603. [torch._C._nn.upsample_linear1d]It is expected output_size equals to 1, but got size 0
 604. [torch._C._nn.upsample_linear1d]Expected out tensor to have dtype float, but got int instead
 605. [torch._C._nn.upsample_linear1d]Trying to resize storage that is not resizable
 606. [torch._C._nn.upsample_nearest1d]Expected static_cast<int64_t>(scale_factors->size()) == spatial_dimensions to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 607. [torch._C._nn.upsample_nearest1d]Non-empty 3D data tensor expected but got a tensor with sizes [8, 0, 7]
 608. [torch._C._nn.upsample_nearest1d]compute_indices_weights_nearest not implemented for 'Int'
 609. [torch._C._nn.upsample_nearest1d]Input and output sizes should be greater than 0, but got input (W: 8) and output (W: -3)
 610. [torch._C._nn.upsample_nearest1d]It is expected output_size equals to 1, but got size 0
 61
 612. [torch._C._nn.upsample_nearest2d]upsample_nearest2d_channels_last not implemented for 'Bool'
 613. [torch._C._nn.upsample_nearest2d]upsample_nearest2d_channels_last not implemented for 'Long'
 614. [torch._C._nn.upsample_nearest2d]Non-empty 4D data tensor expected but got a tensor with sizes [0, 0, 1, 1]
 615. [torch._C._nn.upsample_nearest2d]Input and output sizes should be greater than 0, but got input (H: 1, W: 0) output (H: 1, W: 1)
 616. [torch._C._nn.upsample_nearest2d]upsample_nearest2d_channels_last not implemented for 'Int'
 617. [torch._C._nn.upsample_nearest2d]It is expected input_size equals to 4, but got size 3
 618. [torch._C._nn.upsample_nearest3d]It is expected output_size equals to 3, but got size 0
 619. [torch._C._nn.upsample_nearest3d]Input and output sizes should be greater than 0, but got input (D: 8, H: 2, W: 9) output (D: 5, H: 3, W: -3)
 62
 621. [torch._C._nn.upsample_nearest3d]compute_indices_weights_nearest not implemented for 'Bool'
 622. [torch._C._nn.upsample_nearest3d]compute_indices_weights_nearest not implemented for 'Int'
 623. [torch._C._nn.upsample_trilinear3d]It is expected output_size equals to 3, but got size 0
 624. [torch._C._nn.upsample_trilinear3d]Input and output sizes should be greater than 0, but got input (D: 9, H: 4, W: 2) output (D: 3, H: -2, W: 3)
 625. [torch._C._nn.upsample_trilinear3d]compute_indices_weights_linear not implemented for 'Int'
 626. [torch._C._nn.upsample_trilinear3d]Trying to resize storage that is not resizable
 627. [torch._C._special.special_airy_ai]airy_ai_cpu not implemented for 'Half'
 628. [torch._C._special.special_airy_ai]result type Float can't be cast to the desired output type Int
 629. [torch._C._special.special_airy_ai]Trying to resize storage that is not resizable
 630. [torch._C._special.special_bessel_j0]result type Float can't be cast to the desired output type Int
 631. [torch._C._special.special_bessel_j0]Trying to resize storage that is not resizable
 632. [torch._C._special.special_bessel_j1]result type Float can't be cast to the desired output type Int
 633. [torch._C._special.special_bessel_j1]Trying to resize storage that is not resizable
 634. [torch._C._special.special_bessel_y0]result type Float can't be cast to the desired output type Int
 635. [torch._C._special.special_bessel_y0]Trying to resize storage that is not resizable
 636. [torch._C._special.special_bessel_y1]result type Float can't be cast to the desired output type Int
 637. [torch._C._special.special_bessel_y1]Trying to resize storage that is not resizable
 638. [torch._C._special.special_chebyshev_polynomial_t]The size of tensor a (5) must match the size of tensor b (7) at non-singleton dimension 5
 639. [torch._C._special.special_chebyshev_polynomial_t]Trying to resize storage that is not resizable
 640. [torch._C._special.special_chebyshev_polynomial_u]The size of tensor a (0) must match the size of tensor b (3) at non-singleton dimension 3
 641. [torch._C._special.special_entr]negative dimensions are not allowed
 642. [torch._C._special.special_entr]result type Float can't be cast to the desired output type Long
 643. [torch._C._special.special_entr]Trying to resize storage that is not resizable
 644. [torch._C._special.special_erfcx]erfcx_cpu not implemented for 'Half'
 645. [torch._C._special.special_erfcx]result type Float can't be cast to the desired output type Int
 646. [torch._C._special.special_erfcx]Trying to resize storage that is not resizable
 647. [torch._C._special.special_hermite_polynomial_h]The size of tensor a (8) must match the size of tensor b (9) at non-singleton dimension 6
 648. [torch._C._special.special_hermite_polynomial_h]result type Float can't be cast to the desired output type Int
 649. [torch._C._special.special_hermite_polynomial_h]Trying to resize storage that is not resizable
 650. [torch._C._special.special_hermite_polynomial_h]negative dimensions are not allowed
 651. [torch._C._special.special_hermite_polynomial_he]Trying to resize storage that is not resizable
 652. [torch._C._special.special_hermite_polynomial_he]The size of tensor a (5) must match the size of tensor b (9) at non-singleton dimension 6
 653. [torch._C._special.special_hermite_polynomial_he]result type Float can't be cast to the desired output type Int
 654. [torch._C._special.special_i0e]result type Float can't be cast to the desired output type Int
 655. [torch._C._special.special_i0e]Trying to resize storage that is not resizable
 656. [torch._C._special.special_i1]i1_cpu not implemented for 'Half'
 657. [torch._C._special.special_i1]result type Float can't be cast to the desired output type Int
 658. [torch._C._special.special_i1]Trying to resize storage that is not resizable
 659. [torch._C._special.special_i1e]result type Float can't be cast to the desired output type Int
 660. [torch._C._special.special_i1e]Trying to resize storage that is not resizable
 661. [torch._C._special.special_laguerre_polynomial_l]The size of tensor a (4) must match the size of tensor b (9) at non-singleton dimension 5
 662. [torch._C._special.special_laguerre_polynomial_l]result type Float can't be cast to the desired output type Int
 663. [torch._C._special.special_laguerre_polynomial_l]Trying to resize storage that is not resizable
 664. [torch._C._special.special_log_ndtr]Trying to resize storage that is not resizable
 665. [torch._C._special.special_log_ndtr]result type Float can't be cast to the desired output type Int
 666. [torch._C._special.special_modified_bessel_i0]result type Float can't be cast to the desired output type Int
 667. [torch._C._special.special_modified_bessel_i0]Trying to resize storage that is not resizable
 668. [torch._C._special.special_modified_bessel_i1]result type Float can't be cast to the desired output type Int
 669. [torch._C._special.special_modified_bessel_i1]Trying to resize storage that is not resizable
 670. [torch._C._special.special_modified_bessel_k0]negative dimensions are not allowed
 671. [torch._C._special.special_modified_bessel_k0]result type Float can't be cast to the desired output type Int
 672. [torch._C._special.special_modified_bessel_k0]Trying to resize storage that is not resizable
 673. [torch._C._special.special_modified_bessel_k1]result type Float can't be cast to the desired output type Int
 674. [torch._C._special.special_modified_bessel_k1]Trying to resize storage that is not resizable
 675. [torch._C._special.special_ndtr]result type Float can't be cast to the desired output type Int
 676. [torch._C._special.special_ndtr]Trying to resize storage that is not resizable
 677. [torch._C._special.special_ndtri]result type Float can't be cast to the desired output type Int
 678. [torch._C._special.special_ndtri]Trying to resize storage that is not resizable
 679. [torch._C._special.special_scaled_modified_bessel_k0]result type Float can't be cast to the desired output type Int
 680. [torch._C._special.special_scaled_modified_bessel_k0]Trying to resize storage that is not resizable
 681. [torch._C._special.special_scaled_modified_bessel_k1]result type Float can't be cast to the desired output type Int
 682. [torch._C._special.special_scaled_modified_bessel_k1]Trying to resize storage that is not resizable
 683. [torch._C._special.special_spherical_bessel_j0]spherical_bessel_j0_cpu not implemented for 'Half'
 684. [torch._C._special.special_spherical_bessel_j0]result type Float can't be cast to the desired output type Int
 685. [torch._C._special.special_spherical_bessel_j0]Trying to resize storage that is not resizable
 686. [torch._C._special.special_xlog1py]The size of tensor a (4) must match the size of tensor b (7) at non-singleton dimension 2
 687. [torch._C._special.special_xlog1py]Trying to resize storage that is not resizable
 688. [torch._C._special.special_xlog1py]result type Float can't be cast to the desired output type Int
 689. [torch._C._special.special_zeta]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 3
 690. [torch.abs]Found dtype Float but expected Int
 691. [torch.abs]Trying to resize storage that is not resizable
 692. [torch.acos]result type Float can't be cast to the desired output type Short
 693. [torch.acos]Trying to resize storage that is not resizable
 694. [torch.acosh]result type Float can't be cast to the desired output type Int
 695. [torch.acosh]Trying to resize storage that is not resizable
 696. [torch.adaptive_avg_pool1d]adaptive_avg_pool1d() argument 'output_size' should contain one int (got 2)
 697. [torch.adaptive_avg_pool1d]adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0 but received {1, -1}
 698. [torch.adaptive_avg_pool1d]Expected 2 to 3 dimensions, but got 1-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)
 699. [torch.adaptive_max_pool1d]adaptive_max_pool1d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes [2, 0, 9] with dimension 1 being empty
 700. [torch.adaptive_max_pool1d]adaptive_max_pool2d not implemented for 'Int'
 701. [torch.adaptive_max_pool1d]Trying to create tensor with negative dimension -3: [3, 7, 1, -3]
 702. [torch.adaptive_max_pool1d]Expected 2 to 3 dimensions, but got 7-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_max_pool1d)
 703. [torch.adaptive_max_pool1d]adaptive_max_pool1d() argument 'output_size' should contain one int (got 0)
 704. [torch.add]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 6
 705. [torch.add]result type Float can't be cast to the desired output type Int
 706. [torch.add]Trying to resize storage that is not resizable
 707. [torch.addcdiv]Integer division with addcdiv is no longer supported, and in a future  release addcdiv will perform a true division of tensor1 and tensor2. The historic addcdiv behavior can be implemented as (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for integer inputs and as (input + value * tensor1 / tensor2) for float inputs. The future addcdiv behavior is just the latter implementation: (input + value * tensor1 / tensor2), for all dtypes.
 708. [torch.addcdiv]The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 6
 709. [torch.addcdiv]Trying to resize storage that is not resizable
 710. [torch.addcdiv]result type Float can't be cast to the desired output type Int
 711. [torch.addcmul]The size of tensor a (9) must match the size of tensor b (4) at non-singleton dimension 4
 712. [torch.addcmul]result type Float can't be cast to the desired output type Int
 713. [torch.addcmul]Trying to resize storage that is not resizable
 714. [torch.addmm]mat1 must be a matrix, got 4-D tensor
 715. [torch.addmm]mat1 and mat2 shapes cannot be multiplied (3x6 and 2x8)
 716. [torch.addmm]expand(torch.FloatTensor{[3, 9, 5, 8, 9, 7, 9]}, size=[6, 8]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (7)
 717. [torch.addmv]expected scalar type Float but found Int
 718. [torch.addmv]size mismatch, got input (8), mat (9x5), vec (7)
 719. [torch.addmv]vector + matrix @ vector expected, got 7, 7, 7
 720. [torch.addmv]Dimension specified as 0 but tensor has no dimensions
 721. [torch.addr]The expanded size of the tensor (3) must match the existing size (6) at non-singleton dimension 1.  Target sizes: [8, 3].  Tensor sizes: [6, 6]
 722. [torch.addr]addr: Expected 1-D argument vec1, but got 7-D
 723. [torch.all]Dimension out of range (expected to be in range of [-2, 1], but got 9)
 724. [torch.all]all only supports bool tensor for result, got: Float
 725. [torch.all]Trying to resize storage that is not resizable
 726. [torch.all]negative dimensions are not allowed
 727. [torch.amax]Dimension out of range (expected to be in range of [-7, 6], but got 92)
 728. [torch.amax]Expected the dtype for input and out to match, but got Int for input's dtype and Float for out's dtype.
 729. [torch.amin]Dimension out of range (expected to be in range of [-2, 1], but got 55)
 730. [torch.amin]amin(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
 731. [torch.aminmax]Dimension out of range (expected to be in range of [-2, 1], but got 55)
 732. [torch.angle]result type Float can't be cast to the desired output type Int
 733. [torch.angle]Trying to resize storage that is not resizable
 734. [torch.any]Dimension out of range (expected to be in range of [-2, 1], but got 2)
 735. [torch.any]any only supports bool tensor for result, got: Float
 736. [torch.any]negative dimensions are not allowed
 737. [torch.any]Trying to resize storage that is not resizable
 738. [torch.argmax]argmax(): Expected reduction dim 0 to have non-zero size.
 739. [torch.argmax]Dimension out of range (expected to be in range of [-2, 1], but got 100)
 740. [torch.argmin]Dimension out of range (expected to be in range of [-1, 0], but got 8)
 741. [torch.argmin]Expected out tensor to have dtype long int, but got float instead
 742. [torch.argmin]Trying to resize storage that is not resizable
 743. [torch.argsort]Dimension out of range (expected to be in range of [-2, 1], but got 5)
 744. [torch.asin]result type Float can't be cast to the desired output type Int
 745. [torch.asin]Trying to resize storage that is not resizable
 746. [torch.asinh]result type Float can't be cast to the desired output type Int
 747. [torch.asinh]Trying to resize storage that is not resizable
 748. [torch.atan]result type Float can't be cast to the desired output type Int
 749. [torch.atan]Trying to resize storage that is not resizable
 750. [torch.atan2]The size of tensor a (6) must match the size of tensor b (7) at non-singleton dimension 2
 751. [torch.atan2]result type Float can't be cast to the desired output type Int
 752. [torch.atan2]Trying to resize storage that is not resizable
 753. [torch.atanh]result type Float can't be cast to the desired output type Int
 754. [torch.atanh]Trying to resize storage that is not resizable
 755. [torch.bernoulli]bernoulli_ expects p to be in [0, 1], but got p=3
 756. [torch.binary_cross_entropy_with_logits]output with shape [] doesn't match the broadcast shape [1, 1, 1, 1]
 757. [torch.binary_cross_entropy_with_logits]log_sigmoid_cpu not implemented for 'Int'
 758. [torch.binary_cross_entropy_with_logits]result type Float can't be cast to the desired output type Int
 759. [torch.binary_cross_entropy_with_logits]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 3
 760. [torch.bitwise_and]bitwise_and_cpu not implemented for 'Float'
 761. [torch.bitwise_and]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 2
 762. [torch.bitwise_and]Trying to resize storage that is not resizable
 763. [torch.bitwise_and]result type Float can't be cast to the desired output type Short
 764. [torch.bitwise_and]bitwise_and_cpu not implemented for 'Half'
 765. [torch.bitwise_left_shift]lshift_cpu not implemented for 'Float'
 766. [torch.bitwise_left_shift]The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 5
 767. [torch.bitwise_left_shift]Trying to resize storage that is not resizable
 768. [torch.bitwise_left_shift]lshift_cpu not implemented for 'Half'
 769. [torch.bitwise_not]bitwise_not_cpu not implemented for 'Float'
 770. [torch.bitwise_not]bitwise_not_cpu not implemented for 'Half'
 771. [torch.bitwise_not]Trying to resize storage that is not resizable
 772. [torch.bitwise_not]Found dtype Float but expected Long
 773. [torch.bitwise_or]bitwise_or_cpu not implemented for 'Float'
 774. [torch.bitwise_or]The size of tensor a (4) must match the size of tensor b (7) at non-singleton dimension 2
 775. [torch.bitwise_or]Trying to resize storage that is not resizable
 776. [torch.bitwise_right_shift]rshift_cpu not implemented for 'Float'
 777. [torch.bitwise_right_shift]The size of tensor a (8) must match the size of tensor b (9) at non-singleton dimension 4
 778. [torch.bitwise_right_shift]Trying to resize storage that is not resizable
 779. [torch.bitwise_right_shift]negative dimensions are not allowed
 780. [torch.bitwise_xor]The size of tensor a (3) must match the size of tensor b (9) at non-singleton dimension 4
 781. [torch.bitwise_xor]bitwise_xor_cpu not implemented for 'Float'
 782. [torch.bitwise_xor]bitwise_xor_cpu not implemented for 'Half'
 783. [torch.bitwise_xor]result type Float can't be cast to the desired output type Long
 784. [torch.bitwise_xor]Trying to resize storage that is not resizable
 785. [torch.block_diag]expected Tensor as element 0 in argument 0, but got list
 786. [torch.block_diag]torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 0 has 7 dimensions
 787. [torch.bmm]batch2 must be a 3D tensor
 788. [torch.bmm]Expected size for first two dimensions of batch2 tensor to be: [3, 7] but got: [5, 3].
 789. [torch.bmm]Trying to resize storage that is not resizable
 790. [torch.bmm]Expected out tensor to have dtype float, but got int instead
 791. [torch.broadcast_to]The expanded size of the tensor (1) must match the existing size (4) at non-singleton dimension 8.  Target sizes: [2, 1, 9, 1, 1, 1, 1, 1, 1].  Tensor sizes: [4]
 792. [torch.broadcast_to]numel: integer multiplication overflow
 793. [torch.broadcast_to]expand(torch.FloatTensor{[1, 1]}, size=[]): the number of sizes provided (0) must be greater or equal to the number of dimensions in the tensor (2)
 794. [torch.bucketize]boundaries tensor must be 1 dimension, but got dim(3)
 795. [torch.bucketize]torch.searchsorted(): output tensor's dtype is wrong, it can only be Int(int32) or Long(int64) depending on whether out_int32 flag is True, but we got output tensor's dtype Float and out_int32 flag is False
 796. [torch.bucketize]Trying to resize storage that is not resizable
 797. [torch.cartesian_prod]Expect a 1D vector, but got shape [0, 0]
 798. [torch.cartesian_prod]meshgrid expects a non-empty TensorList
 799. [torch.cat]torch.cat(): expected a non-empty list of Tensors
 800. [torch.cat]Dimension out of range (expected to be in range of [-2, 1], but got 6)
 801. [torch.cat]Name 'AnIx' not found in Tensor[None, None, None, None, None, None, None, None, None].
 802. [torch.cat]Trying to resize storage that is not resizable
 803. [torch.cat]Too large tensor shape: shape = [9, 9, 9, 9, 9, 9, 9, 9, 9]
 804. [torch.ceil]Found dtype Float but expected Int
 805. [torch.ceil]Trying to resize storage that is not resizable
 806. [torch.cholesky]cholesky: A must be batches of square matrices, but they are 1 by 7 matrices
 807. [torch.cholesky_inverse]cholesky_inverse: A must be batches of square matrices, but they are 8 by 6 matrices
 808. [torch.cholesky_inverse]cholesky_inverse: The input tensor A must have at least 2 dimensions.
 809. [torch.cholesky_inverse]Trying to resize storage that is not resizable
 810. [torch.chunk]chunk expects at least a 1-dimensional tensor
 811. [torch.chunk]chunk expects `chunks` to be greater than 0, got: -3
 812. [torch.chunk]Dimension out of range (expected to be in range of [-1, 0], but got -4)
 813. [torch.clamp]The size of tensor a (6) must match the size of tensor b (7) at non-singleton dimension 3
 814. [torch.clamp]Trying to resize storage that is not resizable
 815. [torch.clamp_]The size of tensor a (5) must match the size of tensor b (9) at non-singleton dimension 3
 816. [torch.clamp_max]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 3
 817. [torch.clamp_max]Found dtype Float but expected Int
 818. [torch.clamp_max]Trying to resize storage that is not resizable
 819. [torch.clamp_max]result type Float can't be cast to the desired output type Int
 820. [torch.clamp_max_]result type Float can't be cast to the desired output type Int
 821. [torch.clamp_max_]The size of tensor a (8) must match the size of tensor b (6) at non-singleton dimension 3
 822. [torch.clamp_min]The size of tensor a (6) must match the size of tensor b (7) at non-singleton dimension 3
 823. [torch.clamp_min]Found dtype Float but expected Int
 824. [torch.clamp_min]Trying to resize storage that is not resizable
 825. [torch.clamp_min]result type Float can't be cast to the desired output type Int
 826. [torch.clamp_min_]The size of tensor a (9) must match the size of tensor b (6) at non-singleton dimension 3
 827. [torch.column_stack]column_stack expects a non-empty TensorList
 828. [torch.column_stack]Tensors must have same number of dimensions: got 6 and 2
 829. [torch.combinations]Expect a non-negative number, but got -3
 830. [torch.combinations]Expect a 1D vector, but got shape [6, 6]
 831. [torch.complex]Expected both inputs to be Half, Float or Double tensors but got Float and Int
 832. [torch.complex]The size of tensor a (8) must match the size of tensor b (3) at non-singleton dimension 3
 833. [torch.complex]Expected object of scalar type ComplexFloat but got scalar type Float for argument 'out'
 834. [torch.conj_physical]negative dimensions are not allowed
 835. [torch.conj_physical]Found dtype Int but expected Float
 836. [torch.conj_physical]Trying to resize storage that is not resizable
 837. [torch.constant_pad_nd]Length of pad must be even but instead it equals 1
 838. [torch.copysign]The size of tensor a (7) must match the size of tensor b (8) at non-singleton dimension 2
 839. [torch.copysign]Trying to resize storage that is not resizable
 840. [torch.copysign]result type Float can't be cast to the desired output type Int
 841. [torch.cos]result type Float can't be cast to the desired output type Int
 842. [torch.cos]Trying to resize storage that is not resizable
 843. [torch.cosh]result type Float can't be cast to the desired output type Int
 844. [torch.cosh]Trying to resize storage that is not resizable
 845. [torch.cosine_embedding_loss]The size of tensor a (9) must match the size of tensor b (7) at non-singleton dimension 0
 846. [torch.cosine_embedding_loss]0D target tensor expects 1D input tensors, but found inputs with sizes [7, 8, 8, 9] and [7, 9, 9, 9, 9, 9].
 847. [torch.cosine_embedding_loss]0D or 1D target tensor expected, multi-target not supported
 848. [torch.cosine_similarity]Dimension out of range (expected to be in range of [-7, 6], but got 9)
 849. [torch.cosine_similarity]The size of tensor a (7) must match the size of tensor b (4) at non-singleton dimension 1
 850. [torch.count_nonzero]Dimension out of range (expected to be in range of [-1, 0], but got 33)
 851. [torch.cov]cov(): expected input to have two or fewer dimensions but got an input with 7 dimensions
 852. [torch.cov]cov(): expected fweights to have one or fewer dimensions but got fweights with 2 dimensions
 853. [torch.cross]expected scalar type Float but found Int
 854. [torch.cross]Dimension out of range (expected to be in range of [-3, 2], but got 56)
 855. [torch.cross]The size of tensor a (8) must match the size of tensor b (7) at non-singleton dimension 2
 856. [torch.cross]linalg.cross: inputs dimension 1 must have length 3. Got 3 and 2
 857. [torch.cross]no dimension of size 3 in input
 858. [torch.cross]linalg.cross: inputs must have the same number of dimensions.
 859. [torch.cross]Trying to resize storage that is not resizable
 860. [torch.cross]Expected out tensor to have dtype float, but got int instead
 861. [torch.cross]Dimension specified as 0 but tensor has no dimensions
 862. [torch.cummax]Dimension out of range (expected to be in range of [-7, 6], but got 7)
 863. [torch.cummin]Dimension out of range (expected to be in range of [-7, 6], but got 8)
 864. [torch.cumprod]Dimension out of range (expected to be in range of [-7, 6], but got 8)
 865. [torch.cumprod]Trying to resize storage that is not resizable
 866. [torch.cumsum]Dimension out of range (expected to be in range of [-7, 6], but got 8)
 867. [torch.cumsum]Trying to resize storage that is not resizable
 868. [torch.cumulative_trapezoid]The size of tensor a (5) must match the size of tensor b (0) at non-singleton dimension 0
 869. [torch.cumulative_trapezoid]Dimension out of range (expected to be in range of [-7, 6], but got 8)
 870. [torch.deg2rad]result type Float can't be cast to the desired output type Int
 871. [torch.deg2rad]Trying to resize storage that is not resizable
 872. [torch.diag]diag(): Supports 1D or 2D tensors. Got 7D
 873. [torch.diag]Trying to resize storage that is not resizable
 874. [torch.diag]Expected out tensor to have dtype float, but got int instead
 875. [torch.diag_embed]diagonal dimensions cannot be identical -2, 6
 876. [torch.diag_embed]Dimension out of range (expected to be in range of [-4, 3], but got 9)
 877. [torch.diagonal]diagonal dimensions cannot be identical -3, 0
 878. [torch.diagonal]Dimension out of range (expected to be in range of [-2, 1], but got 8)
 879. [torch.diagonal_copy]diagonal dimensions cannot be identical -1, 2
 880. [torch.diagonal_copy]Dimension out of range (expected to be in range of [-3, 2], but got 6)
 881. [torch.diagonal_copy]Trying to resize storage that is not resizable
 882. [torch.diff]Dimension out of range (expected to be in range of [-2, 1], but got -4)
 883. [torch.diff]diff expects prepend or append to be the same dimension as input
 884. [torch.diff]diff expects the shape of tensor to prepend or append to match that of input except along the differencing dimension; input.size(1) = 7, but got tensor.size(1) = 2
 885. [torch.diff]order must be non-negative but got -4
 886. [torch.diff]diff expects input to be at least one-dimensional
 887. [torch.digamma]result type Float can't be cast to the desired output type Int
 888. [torch.digamma]Trying to resize storage that is not resizable
 889. [torch.div]The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 6
 890. [torch.div]div expected rounding_mode to be one of None, 'trunc', or 'floor' but found 'vmBq'
 891. [torch.div]ZeroDivisionError
 892. [torch.div]result type Float can't be cast to the desired output type Int
 893. [torch.div]Trying to resize storage that is not resizable
 894. [torch.divide]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 3
 895. [torch.divide]div expected rounding_mode to be one of None, 'trunc', or 'floor' but found 'SzmG'
 896. [torch.divide]Trying to resize storage that is not resizable
 897. [torch.divide]result type Float can't be cast to the desired output type Int
 898. [torch.dsplit]number of sections must be larger than 0, got -4
 899. [torch.dsplit]torch.dsplit attempted to split along dimension 2, but the size of the dimension 6 is not divisible by the split_size 5!
 900. [torch.dsplit]torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with 2 dimensions!
 901. [torch.dstack]dstack expects a non-empty TensorList
 902. [torch.eq]The size of tensor a (3) must match the size of tensor b (9) at non-singleton dimension 4
 903. [torch.eq]Trying to resize storage that is not resizable
 904. [torch.erf]result type Float can't be cast to the desired output type Int
 905. [torch.erf]Trying to resize storage that is not resizable
 906. [torch.erfc]result type Float can't be cast to the desired output type Int
 907. [torch.erfc]Trying to resize storage that is not resizable
 908. [torch.erfinv]result type Float can't be cast to the desired output type Int
 909. [torch.erfinv]Trying to resize storage that is not resizable
 910. [torch.exp]result type Float can't be cast to the desired output type Int
 911. [torch.exp]Trying to resize storage that is not resizable
 912. [torch.exp2]result type Float can't be cast to the desired output type Int
 913. [torch.exp2]Trying to resize storage that is not resizable
 914. [torch.expm1]result type Float can't be cast to the desired output type Long
 915. [torch.expm1]Trying to resize storage that is not resizable
 916. [torch.flatten]flatten() has invalid args: start_dim cannot come after end_dim
 917. [torch.flatten]Dimension out of range (expected to be in range of [-2, 1], but got -3)
 918. [torch.flip]Dimension out of range (expected to be in range of [-2, 1], but got 34)
 919. [torch.fliplr]Input must be >= 2-d.
 920. [torch.flipud]Input must be >= 1-d.
 921. [torch.float_power]Trying to resize storage that is not resizable
 922. [torch.float_power]The size of tensor a (3) must match the size of tensor b (9) at non-singleton dimension 1
 923. [torch.float_power]the output given to float_power has dtype Float but the operation's result requires dtype Double
 924. [torch.floor]Found dtype Int but expected Float
 925. [torch.floor]Trying to resize storage that is not resizable
 926. [torch.floor_divide]The size of tensor a (6) must match the size of tensor b (9) at non-singleton dimension 4
 927. [torch.floor_divide]Trying to resize storage that is not resizable
 928. [torch.floor_divide]ZeroDivisionError
 929. [torch.floor_divide]result type Float can't be cast to the desired output type Int
 930. [torch.fmax]The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 6
 931. [torch.fmax]Trying to resize storage that is not resizable
 932. [torch.fmin]The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 4
 933. [torch.fmin]Trying to resize storage that is not resizable
 934. [torch.fmin]result type Float can't be cast to the desired output type Int
 935. [torch.fmod]The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 4
 936. [torch.fmod]Trying to resize storage that is not resizable
 937. [torch.fmod]negative dimensions are not allowed
 938. [torch.fmod]result type Float can't be cast to the desired output type Int
 939. [torch.frac]frac_cpu not implemented for 'Int'
 940. [torch.frac]Found dtype Float but expected Int
 941. [torch.frac]Trying to resize storage that is not resizable
 942. [torch.frexp]torch.frexp() only supports floating-point dtypes
 943. [torch.full_like]Need to provide pin_memory allocator to use pin memory.
 944. [torch.gcd]gcd_cpu not implemented for 'Half'
 945. [torch.gcd]gcd_cpu not implemented for 'Float'
 946. [torch.gcd]The size of tensor a (4) must match the size of tensor b (8) at non-singleton dimension 5
 947. [torch.gcd]Trying to resize storage that is not resizable
 948. [torch.ge]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 3
 949. [torch.ge]negative dimensions are not allowed
 950. [torch.ge]Trying to resize storage that is not resizabl
 952. [torch.geqrf]geqrf_cpu not implemented for 'Bool'
 953. [torch.geqrf]geqrf_cpu not implemented for 'Long'
 954. [torch.geqrf]geqrf_cpu not implemented for 'Int'
 955. [torch.geqrf]torch.geqrf: input must have at least 2 dimensions.
 956. [torch.gradient]torch.gradient only supports edge_order=1 and edge_order=2.
 957. [torch.gradient]Dimension out of range (expected to be in range of [-2, 1], but got 8)
 958. [torch.gt]The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 4
 959. [torch.gt]Trying to resize storage that is not resizable
 960. [torch.hardshrink]hardshrink_cpu not implemented for 'Int'
 961. [torch.hardshrink]hardshrink_cpu not implemented for 'ComplexDouble'
 962. [torch.hardshrink]Found dtype Int but expected Float
 963. [torch.hardshrink]Trying to resize storage that is not resizable
 964. [torch.heaviside]heaviside is not yet implemented for tensors with different dtypes.
 965. [torch.heaviside]The size of tensor a (4) must match the size of tensor b (7) at non-singleton dimension 4
 966. [torch.hinge_embedding_loss]The size of tensor a (7) must match the size of tensor b (9) at non-singleton dimension 
 968. [torch.histc]linspace_cpu not implemented for 'Bool'
 969. [torch.histc]histogram_cpu not implemented for 'Int'
 970. [torch.histc]torch.histogram(): bins must be > 0, but got -1 for dimension 0
 971. [torch.histc]torch.histc: max must be larger than min
 972. [torch.histc]torch.histogram: input tensor and hist tensor should have the same dtype, but got input float and hist int
 973. [torch.histc]Trying to resize storage that is not resizable
 974. [torch.histogram]torch.histogramdd: bins tensor should have one dimension, but got 7 dimensions in the bins tensor for dimension 0
 975. [torch.histogram]torch.histogramdd: if weight tensor is provided it should have the same shape as the input tensor excluding its innermost dimension, but got input with shape [9, 1] and weight with shape [729]
 976. [torch.histogramdd]histogramdd: The size of bins must be equal to the innermost dimension of the input.
 977. [torch.histogramdd]torch.histogramdd: input tensor should have at least 2 dimensions
 978. [torch.hsplit]torch.hsplit attempted to split along dimension 0, but the size of the dimension 8 is not divisible by the split_size 0!
 979. [torch.hsplit]torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with 0 dimensions!
 980. [torch.hstack]hstack expects a non-empty TensorList
 981. [torch.hstack]Tensors must have same number of dimensions: got 6 and 9
 982. [torch.hstack]Trying to resize storage that is not resizable
 983. [torch.hypot]The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 2
 984. [torch.i0]result type Float can't be cast to the desired output type Int
 985. [torch.i0]Trying to resize storage that is not resizable
 986. [torch.igamma]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 2
 987. [torch.igammac]The size of tensor a (8) must match the size of tensor b (9) at non-singleton dimension 6
 988. [torch.igammac]igammac_cpu not implemented for 'Int'
 989. [torch.imag]imag is not implemented for tensors with non-complex dtypes.
 990. [torch.index_put]shape mismatch: value tensor of shape [3, 3, 3, 3] cannot be broadcast to indexing result of shape [1, 1]
 991. [torch.isin]Trying to resize storage that is not resizable
 992. [torch.isin]Expected out tensor to have dtype bool, but got float instead
 993. [torch.isneginf]Trying to resize storage that is not resizable
 994. [torch.isneginf]isneginf does not support non-boolean outputs.
 995. [torch.isposinf]isposinf does not support non-boolean outputs.
 996. [torch.isposinf]Trying to resize storage that is not resizable
 997. [torch.kron]Trying to resize storage that is not resizable
 998. [torch.kthvalue]Dimension out of range (expected to be in range of [-2, 1], but got 9)
 999. [torch.kthvalue]kthvalue(): selected number k out of range for dimension 0
 1000. [torch.lcm]lcm_cpu not implemented for 'Float'
 1001. [torch.lcm]The size of tensor a (7) must match the size of tensor b (8) at non-singleton dimension 5
 1002. [torch.lcm]Trying to resize storage that is not resizable
 1003. [torch.lcm]lcm_cpu not implemented for 'Half'
 1004. [torch.ldexp]The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 3
 1005. [torch.ldexp]result type Float can't be cast to the desired output type Int
 1006. [torch.ldexp]Trying to resize storage that is not resizable
 1007. [torch.le]The size of tensor a (9) must match the size of tensor b (7) at non-singleton dimension 3
 1008. [torch.le]Trying to resize storage that is not resizable
 1009. [torch.lerp]The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 2
 1010. [torch.lerp]Trying to resize storage that is not resizable
 1011. [torch.lgamma]result type Float can't be cast to the desired output type Int
 1012. [torch.lgamma]Trying to resize storage that is not resizable
 1013. [torch.log]result type Float can't be cast to the desired output type Int
 1014. [torch.log]Trying to resize storage that is not resizable
 1015. [torch.log10]Trying to resize storage that is not resizable
 1016. [torch.log10]result type Float can't be cast to the desired output type Int
 1017. [torch.log1p]result type Float can't be cast to the desired output type Int
 1018. [torch.log1p]Trying to resize storage that is not resizable
 1019. [torch.log2]result type Float can't be cast to the desired output type Int
 1020. [torch.log2]Trying to resize storage that is not resizabl
 1022. [torch.log_softmax]log_softmax_lastdim_kernel_impl not implemented for 'Bool'
 1023. [torch.log_softmax]log_softmax_lastdim_kernel_impl not implemented for 'Int'
 1024. [torch.log_softmax]Dimension out of range (expected to be in range of [-1, 0], but got 1)
 1025. [torch.log_softmax]Expected out tensor to have dtype float, but got int instead
 1026. [torch.log_softmax]Trying to resize storage that is not resizable
 1027. [torch.logaddexp]The size of tensor a (9) must match the size of tensor b (3) at non-singleton dimension 6
 1028. [torch.logaddexp]Trying to resize storage that is not resizable
 1029. [torch.logaddexp]result type Float can't be cast to the desired output type Int
 1030. [torch.logaddexp]logaddexp_cpu not implemented for 'Short'
 1031. [torch.logaddexp2]The size of tensor a (4) must match the size of tensor b (9) at non-singleton dimension 4
 1032. [torch.logaddexp2]logaddexp2_cpu not implemented for 'Int'
 1033. [torch.logaddexp2]logaddexp2_cpu not implemented for 'Boo
 1035. [torch.logaddexp2]Trying to resize storage that is not resizable
 1036. [torch.logaddexp2]result type Float can't be cast to the desired output type Int
 1037. [torch.logcumsumexp]negative dimensions are not allowed
 1038. [torch.logcumsumexp]logcumsumexp_out_cpu not implemented for 'Int'
 1039. [torch.logcumsumexp]Dimension out of range (expected to be in range of [-2, 1], but got 15)
 1040. [torch.logcumsumexp]Trying to resize storage that is not resizabl
 1042. [torch.logcumsumexp]logcumsumexp_out_cpu not implemented for 'Bool'
 1043. [torch.logcumsumexp]expected scalar_type Float but found Int
 1044. [torch.logdet]logdet: A must be batches of square matrices, but they are 6 by 7 matrices
 1045. [torch.logdet]logdet: The input tensor A must have at least 2 dimensions.
 1046. [torch.logical_and]The size of tensor a (8) must match the size of tensor b (3) at non-singleton dimension 4
 1047. [torch.logical_and]Trying to resize storage that is not resizable
 1048. [torch.logical_not]Trying to resize storage that is not resizable
 1049. [torch.logical_or]The size of tensor a (8) must match the size of tensor b (2) at non-singleton dimension 4
 1050. [torch.logical_or]Trying to resize storage that is not resizable
 1051. [torch.logical_xor]The size of tensor a (8) must match the size of tensor b (7) at non-singleton dimension 2
 1052. [torch.logical_xor]Trying to resize storage that is not resizable
 1053. [torch.logit]result type Float can't be cast to the desired output type Int
 1054. [torch.logit]Trying to resize storage that is not resizable
 0. [torch.Tensor.lshift]lshift_cpu not implemented for 'Half'
 1. [torch.Tensor.or]bitwise_or_cpu not implemented for 'Half'
 2. [torch.Tensor.add_]result type Float can't be cast to the desired output type Int
 3. [torch.Tensor.addbmm]expand(torch.FloatTensor{[1, 1, 6, 1, 1, 1, 1]}, size=[1, 1]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (7)
 4. [torch.Tensor.addcdiv_]result type Float can't be cast to the desired output type Int
 5. [torch.Tensor.addmm]addmm_impl_cpu_ not implemented for 'Bool'
 6. [torch.Tensor.addmv]expected scalar type Float but found Int
 7. [torch.Tensor.addmv_]expected scalar type Int but found Float
 8. [torch.Tensor.amax]amax(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
 9. [torch.Tensor.amax]dim 6 appears multiple times in the list of dims
 10. [torch.Tensor.amax]amax(): Expected reduction dim 0 to have non-zero size.
 11. [torch.Tensor.amin]amin(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
 12. [torch.Tensor.amin]dim 5 appears multiple times in the list of dims
 13. [torch.Tensor.amin]amin(): Expected reduction dim 1 to have non-zero size.
 14. [torch.Tensor.aminmax]aminmax: Expected reduction dim 0 to have non-zero size.
 15. [torch.Tensor.as_strided]Tensor: invalid storage offset -2
 16. [torch.Tensor.as_strided]setStorage: sizes [], strides [], storage offset 94, and itemsize 4 requiring a storage size of 380 are out of bounds for storage of size 4
 17. [torch.Tensor.as_strided_]Tensor: invalid storage offset -4
 18. [torch.Tensor.as_strided_]setStorage: sizes [], strides [], storage offset 45, and itemsize 4 requiring a storage size of 184 are out of bounds for storage of size 168
 19. [torch.Tensor.asin_]result type Float can't be cast to the desired output type Int
 20. [torch.Tensor.atan_]result type Float can't be cast to the desired output type Int
 21. [torch.Tensor.bitwise_and]bitwise_and_cpu not implemented for 'Half'
 22. [torch.Tensor.bitwise_and]bitwise_and_cpu not implemented for 'Double'
 23. [torch.Tensor.bitwise_not_]bitwise_not_cpu not implemented for 'Half'
 24. [torch.Tensor.bitwise_or]negative dimensions are not allowed
 25. [torch.Tensor.broadcast_to]The expanded size of the tensor (2) must match the existing size (7) at non-singleton dimension 1.  Target sizes: [3, 2, 3, 3, 4, 1, 4].  Tensor sizes: [1, 7, 3, 3, 1, 1, 1]
 26. [torch.Tensor.cholesky]cholesky: A must be batches of square matrices, but they are 2 by 3 matrices
 27. [torch.Tensor.cholesky]cholesky: The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).
 28. [torch.Tensor.cholesky]cholesky: The input tensor A must have at least 2 dimensions.
 29. [torch.Tensor.cholesky]cholesky_cpu not implemented for 'Int'
 30. [torch.Tensor.cholesky_inverse]cholesky_inverse: A must be batches of square matrices, but they are 6 by 4 matrices
 31. [torch.Tensor.cholesky_inverse]cholesky_inverse: The input tensor A must have at least 2 dimensions.
 32. [torch.Tensor.cholesky_inverse]cholesky_inverse_out_cpu not implemented for 'Int'
 33. [torch.Tensor.cholesky_solve]Expected b and A to have the same dtype, but found b of type Float and A of type Int instead.
 34. [torch.Tensor.cholesky_solve]cholesky_solve_cpu not implemented for 'Int'
 35. [torch.Tensor.clamp]torch.clamp: At least one of 'min' or 'max' must not be None
 36. [torch.Tensor.clamp_]torch.clamp: At least one of 'min' or 'max' must not be None
 37. [torch.Tensor.corrcoef]corrcoef(): expected input to have two or fewer dimensions but got an input with 7 dimensions
 38. [torch.Tensor.cos_]result type Float can't be cast to the desired output type Int
 39. [torch.Tensor.cosh_]result type Float can't be cast to the desired output type Int
 40. [torch.Tensor.count_nonzero]dim 5 appears multiple times in the list of dims
 41. [torch.Tensor.cov]cov(): aweights cannot be negative
 42. [torch.Tensor.cov]cov(): expected aweights to have floating point dtype but got aweights with Int dtype
 43. [torch.Tensor.cov]cov(): weights sum to zero, can't be normalized
 44. [torch.Tensor.det]linalg.det: Expected a floating point or complex tensor as input. Got Int
 45. [torch.Tensor.diagonal]negative dimensions are not allowed
 46. [torch.Tensor.digamma_]result type Float can't be cast to the desired output type Int
 47. [torch.Tensor.dist]linalg.vector_norm cannot compute the -2 norm on an empty tensor because the operation does not have an identity
 48. [torch.Tensor.div_]result type Float can't be cast to the desired output type Int
 49. [torch.Tensor.divide_]result type Float can't be cast to the desired output type Int
 50. [torch.Tensor.divide_]The size of tensor a (7) must match the size of tensor b (3) at non-singleton dimension 2
 51. [torch.Tensor.dsplit]torch.dsplit attempted to split along dimension 2, but the size of the dimension 1 is not divisible by the split_size 4!
 52. [torch.Tensor.erf_]result type Float can't be cast to the desired output type Int
 53. [torch.Tensor.erfc_]result type Float can't be cast to the desired output type Int
 54. [torch.Tensor.erfinv_]result type Float can't be cast to the desired output type Int
 55. [torch.Tensor.exp_]result type Float can't be cast to the desired output type Int
 56. [torch.Tensor.expand]The expanded size of the tensor (4) must match the existing size (3) at non-singleton dimension 6.  Target sizes: [3, 7, 7, 6, 3, 1, 4].  Tensor sizes: [4, 3, 4, 2, 7, 3, 3]
 57. [torch.Tensor.expm1_]result type Float can't be cast to the desired output type Int
 58. [torch.Tensor.flatten]negative dimensions are not allowed
 59. [torch.Tensor.fliplr]Input must be >= 2-d.
 60. [torch.Tensor.flipud]Input must be >= 1-d.
 61. [torch.Tensor.float_power]negative dimensions are not allowed
 62. [torch.Tensor.frac]frac_cpu not implemented for 'Int'
 63. [torch.Tensor.frac_]frac_cpu not implemented for 'Int'
 64. [torch.Tensor.frexp]torch.frexp() only supports floating-point dtypes
 65. [torch.Tensor.hardshrink]hardshrink_cpu not implemented for 'Bool'
 66. [torch.Tensor.heaviside_]The size of tensor a (6) must match the size of tensor b (3) at non-singleton dimension 6
 67. [torch.Tensor.index_select]INDICES element is out of DATA bounds, id=-549749 axis_dim=6
 68. [torch.Tensor.lgamma_]result type Float can't be cast to the desired output type Int
 69. [torch.Tensor.log10_]result type Float can't be cast to the desired output type Int
 70. [torch.Tensor.log1p_]result type Float can't be cast to the desired output type Int
 71. [torch.Tensor.log2_]result type Float can't be cast to the desired output type Int
 72. [torch.Tensor.log_]result type Float can't be cast to the desired output type Int
 73. [torch.Tensor.log_softmax]negative dimensions are not allowed
 74. [torch.Tensor.masked_fill]The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 6
 75. [torch.Tensor.masked_scatter]negative dimensions are not allowed
 76. [torch.Tensor.masked_scatter]Number of elements of source < number of ones in mask
 77. [torch.Tensor.matmul]Expected size for first two dimensions of batch2 tensor to be: [2016, 6] but got: [2016, 2].
 78. [torch.Tensor.matmul]dot : expected both vectors to have same dtype, but found Float and Int
 79. [torch.Tensor.matrix_exp]linalg.matrix_exp: A must be batches of square matrices, but they are 2 by 3 matrices
 80. [torch.Tensor.matrix_exp]linalg.matrix_exp: The input tensor A must have at least 2 dimensions.
 81. [torch.Tensor.matrix_exp]linalg.matrix_exp: Expected a floating point or complex tensor as input. Got Int
 82. [torch.Tensor.matrix_power]linalg.matrix_power: A must be batches of square matrices, but they are 7 by 2 matrices
 83. [torch.Tensor.matrix_power]linalg.matrix_power: The input tensor A must have at least 2 dimensions.
 84. [torch.Tensor.matrix_power]linalg.inv: Expected a floating point or complex tensor as input. Got Int
 85. [torch.Tensor.mean]mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Int
 86. [torch.Tensor.min]The size of tensor a (3) must match the size of tensor b (6) at non-singleton dimension 4
 87. [torch.Tensor.mode]negative dimensions are not allowed
 88. [torch.Tensor.nanmean]Dimension out of range (expected to be in range of [-1, 0], but got 62)
 89. [torch.Tensor.nanmean]nansum does not support complex inputs
 90. [torch.Tensor.new_ones][enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 2313851904000 bytes. Error code 12 (Cannot allocate memory)
 91. [torch.Tensor.new_zeros][enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 2197413036000 bytes. Error code 12 (Cannot allocate memory)
 92. [torch.Tensor.permute]permute(): duplicate dims are not allowed.
 93. [torch.Tensor.permute]negative dimensions are not allowed
 95. [torch.Tensor.repeat][enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 1550224166512 bytes. Error code 12 (Cannot allocate memory)
 96. [torch.Tensor.reshape]shape '[]' is invalid for input of size 720
 97. [torch.Tensor.rot90]Rotation dim0 out of range, dim0 = 7
 98. [torch.Tensor.rot90]expected rotation dims to be different, but got dim0 = 2 and dim1 = 2
 99. [torch.Tensor.scatter_]index 690975 is out of bounds for dimension 5 with size 7
 100. [torch.Tensor.scatter_]scatter(): Expected self.dtype to be equal to src.dtype
 101. [torch.Tensor.scatter_add_]negative dimensions are not allowed
 102. [torch.Tensor.scatter_add_]scatter(): Expected self.dtype to be equal to src.dtype
 103. [torch.Tensor.scatter_reduce_]index -922995 is out of bounds for dimension 0 with size 7
 104. [torch.Tensor.scatter_reduce_]scatter(): Expected self.dtype to be equal to src.dtype
 105. [torch.Tensor.scatter_reduce_]negative dimensions are not allowed
 106. [torch.Tensor.softmax]softmax_kernel_impl not implemented for 'Int'
 107. [torch.Tensor.sort]negative dimensions are not allowed
 108. [torch.Tensor.sub_]output with shape [] doesn't match the broadcast shape [1, 2, 6, 5, 1, 7, 1]
 109. [torch.Tensor.sub_]result type Float can't be cast to the desired output type Int
 110. [torch.Tensor.sum]Dimension out of range (expected to be in range of [-7, 6], but got 52)
 111. [torch.Tensor.unique_consecutive]There are 0 sized dimensions, and they aren't selected, so unique cannot be applied
 112. [torch.Tensor.view]shape '[]' is invalid for input of size 16464
 113. [torch._C._fft.fft_fft]The input size 0, plus negative padding 0 and 0 resulted in a negative output size, which is invalid. Check dimension 2 of your input.
 114. [torch._C._fft.fft_fft2]When given, dim and shape arguments must have the same length
 115. [torch._C._fft.fft_fft2]FFT dims must be unique
 116. [torch._C._fft.fft_fft2]Dimension specified as 1 but tensor has no dimensions
 117. [torch._C._fft.fft_fft2]Dimension out of range (expected to be in range of [-1, 0], but got -2)
 118. [torch._C._fft.fft_fft2]Invalid normalization mode: lhFA
 119. [torch._C._fft.fft_fft2]negative dimensions are not allowed
 120. [torch._C._fft.fft_fftn]Dimension out of range (expected to be in range of [-7, 6], but got 49)
 121. [torch._C._fft.fft_fftn]Invalid number of data points (0) specified
 122. [torch._C._fft.fft_fftn]Got shape with 3 values but input tensor only has 2 dimensions.
 123. [torch._C._fft.fft_fftn]When given, dim and shape arguments must have the same length
 124. [torch._C._fft.fft_fftn]Dimension specified as 37 but tensor has no dimensions
 125. [torch._C._fft.fft_fftn][enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 25702536592656 bytes. Error code 12 (Cannot allocate memory)
 126. [torch._C._fft.fft_fftn]negative dimensions are not allowed
 127. [torch._C._fft.fft_fftn]The input size 0, plus negative padding 0 and 0 resulted in a negative output size, which is invalid. Check dimension 0 of your input.
 128. [torch._C._fft.fft_fftshift]`shifts` required
 129. [torch._C._fft.fft_fftshift]Dimension out of range (expected to be in range of [-7, 6], but got 43)
 130. [torch._C._fft.fft_hfft]Invalid normalization mode: mOjq
 131. [torch._C._fft.fft_hfft]Dimension specified as 0 but tensor has no dimensions
 132. [torch._C._fft.fft_hfft]MKL FFT error: Intel MKL DFTI ERROR: Inconsistent configuration parameters
 133. [torch._C._fft.fft_hfft]The input size 0, plus negative padding 0 and 0 resulted in a negative output size, which is invalid. Check dimension 5 of your input.
 134. [torch._C._fft.fft_hfft]negative dimensions are not allowed
 135. [torch._C._fft.fft_hfft]Invalid number of data points (-1) specified
 136. [torch._C._fft.fft_hfft]hfft expects a floating point output tensor, but got Int
 137. [torch._C._fft.fft_hfft2]When given, dim and shape arguments must have the same length
 138. [torch._C._fft.fft_hfft2]Invalid normalization mode: uXfZ
 139. [torch._C._fft.fft_hfft2]FFT dims must be unique
 140. [torch._C._fft.fft_hfft2]Invalid number of data points (0) specified
 141. [torch._C._fft.fft_hfftn]Dimension out of range (expected to be in range of [-1, 0], but got 61)
 142. [torch._C._fft.fft_hfftn]Got shape with 2 values but input tensor only has 1 dimensions.
 143. [torch._C._fft.fft_hfftn]When given, dim and shape arguments must have the same length
 144. [torch._C._fft.fft_hfftn]Invalid number of data points (0) specified
 145. [torch._C._fft.fft_hfftn][enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 1464489600000 bytes. Error code 12 (Cannot allocate memory)
 146. [torch._C._fft.fft_hfftn]hfftn expects a floating point output tensor, but got Int
 147. [torch._C._fft.fft_ifft]negative dimensions are not allowed
 148. [torch._C._fft.fft_ifft]Invalid number of data points (-3) specified
 149. [torch._C._fft.fft_ifft2]FFT dims must be unique
 150. [torch._C._fft.fft_ifft2]When given, dim and shape arguments must have the same length
 151. [torch._C._fft.fft_ifft2]negative dimensions are not allowed
 15
 153. [torch._C._fft.fft_ifft2]Dimension out of range (expected to be in range of [-1, 0], but got -2)
 154. [torch._C._fft.fft_ifftn]Dimension specified as 6 but tensor has no dimensions
 155. [torch._C._fft.fft_ifftn]Dimension out of range (expected to be in range of [-2, 1], but got 96)
 156. [torch._C._fft.fft_ifftn]Got shape with 2 values but input tensor only has 0 dimensions.
 157. [torch._C._fft.fft_ifftn]When given, dim and shape arguments must have the same length
 158. [torch._C._fft.fft_ifftn]The input size 0, plus negative padding 0 and 0 resulted in a negative output size, which is invalid. Check dimension 4 of your input.
 159. [torch._C._fft.fft_ifftn]Invalid number of data points (0) specified
 160. [torch._C._fft.fft_ifftn]Trying to resize storage that is not resizable
 161. [torch._C._fft.fft_ifftshift]`shifts` required
 162. [torch._C._fft.fft_ifftshift]Dimension out of range (expected to be in range of [-7, 6], but got 53)
 163. [torch._C._fft.fft_ihfft]Dimension specified as 0 but tensor has no dimensions
 164. [torch._C._fft.fft_ihfft]Invalid number of data points (-3) specified
 165. [torch._C._fft.fft_ihfft]negative dimensions are not allowed
 166. [torch._C._fft.fft_ihfft]MKL FFT error: Intel MKL DFTI ERROR: Inconsistent configuration parameters
 167. [torch._C._fft.fft_ihfft]The input size 0, plus negative padding 0 and 0 resulted in a negative output size, which is invalid. Check dimension 0 of your input.
 168. [torch._C._fft.fft_ihfft]Found dtype ComplexDouble but expected ComplexFloat
 169. [torch._C._fft.fft_ihfft2]FFT dims must be unique
 170. [torch._C._fft.fft_ihfftn]Dimension out of range (expected to be in range of [-1, 0], but got 81)
 171. [torch._C._fft.fft_ihfftn]Got shape with 3 values but input tensor only has 1 dimensions.
 172. [torch._C._fft.fft_ihfftn]Invalid number of data points (0) specified
 173. [torch._C._fft.fft_ihfftn]When given, dim and shape arguments must have the same length
 174. [torch._C._fft.fft_ihfftn][enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 1854998179664 bytes. Error code 12 (Cannot allocate memory)
 175. [torch._C._fft.fft_ihfftn]The input size 0, plus negative padding 0 and 0 resulted in a negative output size, which is invalid. Check dimension 1 of your input.
 176. [torch._C._fft.fft_ihfftn]Trying to resize storage that is not resizable
 177. [torch._C._fft.fft_ihfftn]Found dtype ComplexFloat but expected ComplexDouble
 178. [torch._C._fft.fft_irfft]MKL FFT error: Intel MKL DFTI ERROR: Inconsistent configuration parameters
 179. [torch._C._fft.fft_irfft]Unsupported dtype Half
 180. [torch._C._fft.fft_irfft2]FFT dims must be unique
 181. [torch._C._fft.fft_irfft2]When given, dim and shape arguments must have the same length
 182. [torch._C._fft.fft_irfft2]Invalid normalization mode: zXSf
 183. [torch._C._fft.fft_irfft2]Invalid number of data points (0) specified
 184. [torch._C._fft.fft_irfft2]Trying to resize storage that is not resizable
 185. [torch._C._fft.fft_irfft2]MKL FFT error: Intel MKL DFTI ERROR: Inconsistent configuration parameters
 186. [torch._C._fft.fft_irfftn]irfftn must transform at least one axis
 187. [torch._C._fft.fft_irfftn]Invalid number of data points (0) specified
 188. [torch._C._fft.fft_irfftn]Invalid normalization mode: oIFK
 189. [torch._C._fft.fft_irfftn]Dimension out of range (expected to be in range of [-6, 5], but got 65)
 190. [torch._C._fft.fft_irfftn]When given, dim and shape arguments must have the same length
 191. [torch._C._fft.fft_irfftn]Dimension specified as 52 but tensor has no dimensions
 192. [torch._C._fft.fft_irfftn]irfftn expects a floating point output tensor, but got Int
 193. [torch._C._fft.fft_rfft]Dimension specified as 0 but tensor has no dimensions
 194. [torch._C._fft.fft_rfft]Invalid number of data points (-4) specified
 195. [torch._C._fft.fft_rfft]MKL FFT error: Intel MKL DFTI ERROR: Inconsistent configuration parameters
 196. [torch._C._fft.fft_rfft]negative dimensions are not allowed
 197. [torch._C._fft.fft_rfft2]FFT dims must be unique
 198. [torch._C._fft.fft_rfft2]rfftn expects a real-valued input tensor, but got ComplexDouble
 199. [torch._C._fft.fft_rfft2]negative dimensions are not allowed
 200. [torch._C._fft.fft_rfftn]Dimension out of range (expected to be in range of [-7, 6], but got 54)
 201. [torch._C._fft.fft_rfftn]Got shape with 6 values but input tensor only has 1 dimensions.
 202. [torch._C._fft.fft_rfftn]When given, dim and shape arguments must have the same length
 203. [torch._C._fft.fft_rfftn]Invalid number of data points (0) specified
 204. [torch._C._fft.fft_rfftn]rfftn must transform at least one axis
 205. [torch._C._linalg.linalg_cholesky]linalg.cholesky: A must be batches of square matrices, but they are 3 by 1 matrices
 206. [torch._C._linalg.linalg_cholesky]linalg.cholesky: (Batch element 0): The factorization could not be completed because the input is not positive-definite (the leading minor of order 2 is not positive-definite).
 207. [torch._C._linalg.linalg_cholesky]linalg.cholesky: The input tensor A must have at least 2 dimensions.
 208. [torch._C._linalg.linalg_cholesky]linalg.cholesky: Expected a floating point or complex tensor as input. Got Int
 209. [torch._C._linalg.linalg_cholesky]negative dimensions are not allowed
 210. [torch._C._linalg.linalg_cholesky]Expected out tensor to have dtype float, but got int instead
 211. [torch._C._linalg.linalg_cholesky_ex]linalg.cholesky: A must be batches of square matrices, but they are 2 by 6 matrices
 212. [torch._C._linalg.linalg_cholesky_ex]linalg.cholesky_ex: (Batch element 1): The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).
 213. [torch._C._linalg.linalg_cholesky_ex]linalg.cholesky: The input tensor A must have at least 2 dimensions.
 214. [torch._C._linalg.linalg_cholesky_ex]linalg.cholesky: Expected a floating point or complex tensor as input. Got Int
 215. [torch._C._linalg.linalg_cond]linalg.cond(ord=1): A must be batches of square matrices, but they are 4 by 3 matrices
 216. [torch._C._linalg.linalg_cond]linalg.cond(ord=fro): The input tensor A must have at least 2 dimensions.
 217. [torch._C._linalg.linalg_cond]linalg.cond: Expected result to be safely castable from Float dtype, but got result with dtype Int
 218. [torch._C._linalg.linalg_cond]Trying to resize storage that is not resizable
 219. [torch._C._linalg.linalg_cond]negative dimensions are not allowed
 220. [torch._C._linalg.linalg_eig]linalg.eig: A must be batches of square matrices, but they are 3 by 5 matrices
 221. [torch._C._linalg.linalg_eig]linalg.eig: The input tensor A must have at least 2 dimensions.
 222. [torch._C._linalg.linalg_eig]Unknown Complex ScalarType for Int
 223. [torch._C._linalg.linalg_eigh]linalg.eigh: A must be batches of square matrices, but they are 5 by 6 matrices
 224. [torch._C._linalg.linalg_eigh]linalg.eigh: The input tensor A must have at least 2 dimensions.
 225. [torch._C._linalg.linalg_eigh]linalg_eigh_cpu not implemented for 'Int'
 226. [torch._C._linalg.linalg_eigvals]Unknown Complex ScalarType for Int
 227. [torch._C._linalg.linalg_eigvals]negative dimensions are not allowed
 228. [torch._C._linalg.linalg_eigvals]Trying to resize storage that is not resizable
 229. [torch._C._linalg.linalg_eigvalsh]linalg.eigh: The input tensor A must have at least 2 dimensions.
 230. [torch._C._linalg.linalg_eigvalsh]linalg.eigh: A must be batches of square matrices, but they are 3 by 7 matrices
 231. [torch._C._linalg.linalg_eigvalsh]linalg_eigh_cpu not implemented for 'Int'
 232. [torch._C._linalg.linalg_inv]linalg.inv: A must be batches of square matrices, but they are 7 by 1 matrices
 233. [torch._C._linalg.linalg_inv]Trying to resize storage that is not resizable
 234. [torch._C._linalg.linalg_inv]linalg.inv: The input tensor A must have at least 2 dimensions.
 235. [torch._C._linalg.linalg_inv]Expected out tensor to have dtype float, but got int instead
 236. [torch._C._linalg.linalg_inv]linalg.inv: Expected a floating point or complex tensor as input. Got Int
 237. [torch._C._linalg.linalg_inv_ex]linalg.inv: Expected a floating point or complex tensor as input. Got Int
 238. [torch._C._linalg.linalg_ldl_factor]torch.linalg.ldl_factor_ex: Expected a floating point or complex tensor as input. Got Int
 239. [torch._C._linalg.linalg_ldl_factor_ex]torch.linalg.ldl_factor_ex: Expected a floating point or complex tensor as input. Got Int
 240. [torch._C._linalg.linalg_lstsq]This function doesn't handle types other than float and double
 241. [torch._C._linalg.linalg_lstsq]linalg_lstsq_cpu not implemented for 'Int'
 242. [torch._C._linalg.linalg_lstsq]linalg_lstsq() got an unexpected keyword argument 'solution'
 243. [torch._C._linalg.linalg_lu_factor]lu_cpu not implemented for 'Long'
 244. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm: dims must be different. Got (1, 1)
 245. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm: Expected a floating point or complex tensor as input. Got Int
 246. [torch._C._linalg.linalg_matrix_norm]amax(): Expected reduction dim 5 to have non-zero size.
 247. [torch._C._linalg.linalg_matrix_norm]Trying to resize storage that is not resizable
 248. [torch._C._linalg.linalg_matrix_norm]linalg.matrix_norm expected out tensor dtype Float but got: Int
 249. [torch._C._linalg.linalg_matrix_power]linalg.matrix_power: A must be batches of square matrices, but they are 7 by 5 matrices
 250. [torch._C._linalg.linalg_matrix_power]linalg.inv: Expected a floating point or complex tensor as input. Got Int
 251. [torch._C._linalg.linalg_matrix_rank]linalg.eigh: A must be batches of square matrices, but they are 4 by 6 matrices
 252. [torch._C._linalg.linalg_matrix_rank]This function doesn't handle types other than float and double
 253. [torch._C._linalg.linalg_matrix_rank]linalg.svd: Expected a floating point or complex tensor as input. Got Int
 254. [torch._C._linalg.linalg_matrix_rank]linalg_eigh_cpu not implemented for 'Int'
 255. [torch._C._linalg.linalg_norm]linalg.norm: If dim is specified, it must be of length 1 or 2. Got []
 256. [torch._C._linalg.linalg_norm]linalg.vector_norm: Expected a floating point or complex tensor as input. Got Int
 257. [torch._C._linalg.linalg_norm]Dimension out of range (expected to be in range of [-2, 1], but got 27)
 258. [torch._C._linalg.linalg_norm]linalg.matrix_norm: dim must be a 2-tuple. Got 62
 259. [torch._C._linalg.linalg_norm]linalg.matrix_norm: Order 84 not supported.
 260. [torch._C._linalg.linalg_norm]linalg.matrix_norm: The input tensor A must have at least 2 dimensions.
 261. [torch._C._linalg.linalg_pinv]linalg.eigh: A must be batches of square matrices, but they are 1 by 4 matrices
 262. [torch._C._linalg.linalg_pinv]negative dimensions are not allowed
 26
 264. [torch._C._linalg.linalg_solve]Expected out tensor to have dtype float, but got int instead
 265. [torch._C._linalg.linalg_solve_ex]linalg.solve: Expected a floating point or complex tensor as input. Got Int
 266. [torch._C._linalg.linalg_solve_ex]linalg.solve: Expected A and B to have the same dtype, but found A of type Float and B of type Int instead
 267. [torch._C._linalg.linalg_solve_triangular]linalg.solve_triangular: A must be batches of square matrices, but they are 4 by 2 matrices
 268. [torch._C._linalg.linalg_solve_triangular]Trying to resize storage that is not resizable
 269. [torch._C._linalg.linalg_solve_triangular]The size of tensor a (6) must match the size of tensor b (5) at non-singleton dimension 3
 270. [torch._C._linalg.linalg_solve_triangular]triangular_solve_cpu not implemented for 'Int'
 271. [torch._C._linalg.linalg_svd]linalg.svd: The input tensor A must have at least 2 dimensions.
 272. [torch._C._linalg.linalg_svd]linalg.svd: Expected a floating point or complex tensor as input. Got Int
 273. [torch._C._linalg.linalg_svdvals]torch.linalg.svd: keyword argument `driver=` is only supported on CUDA inputs with cuSOLVER backend.
 274. [torch._C._linalg.linalg_svdvals]linalg.svd: The input tensor A must have at least 2 dimensions.
 275. [torch._C._linalg.linalg_svdvals]linalg.svd: Expected a floating point or complex tensor as input. Got Int
 276. [torch._C._linalg.linalg_svdvals]Trying to resize storage that is not resizable
 277. [torch._C._linalg.linalg_svdvals]Expected out tensor to have dtype float, but got int instead
 278. [torch._C._linalg.linalg_tensorinv]Expected self to satisfy the requirement prod(self.shape[ind:]) == prod(self.shape[:ind]), but got 5 != 4
 279. [torch._C._linalg.linalg_tensorinv]linalg.inv: Expected a floating point or complex tensor as input. Got Int
 280. [torch._C._linalg.linalg_tensorinv]tensorinv: Expected result to be safely castable from Float dtype, but got result with dtype Int
 281. [torch._C._linalg.linalg_tensorinv]Trying to resize storage that is not resizable
 282. [torch._C._linalg.linalg_tensorsolve]Expected self to satisfy the requirement prod(self.shape[other.ndim:]) == prod(self.shape[:other.ndim]), but got 1 != 62500
 283. [torch._C._linalg.linalg_tensorsolve]Dimension out of range (expected to be in range of [-7, 6], but got 62)
 284. [torch._C._linalg.linalg_tensorsolve]linalg.solve: Expected A and B to have the same dtype, but found A of type Float and B of type Int instead
 285. [torch._C._linalg.linalg_tensorsolve]linalg.solve: Expected a floating point or complex tensor as input. Got Int
 286. [torch._C._linalg.linalg_tensorsolve]tensorsolve: Expected result to be safely castable from Float dtype, but got result with dtype Int
 287. [torch._C._linalg.linalg_tensorsolve]Trying to resize storage that is not resizable
 288. [torch._C._linalg.linalg_vecdot]negative dimensions are not allowed
 289. [torch._C._linalg.linalg_vector_norm]Dimension out of range (expected to be in range of [-1, 0], but got 68)
 290. [torch._C._linalg.linalg_vector_norm]negative dimensions are not allowed
 291. [torch._C._nn.adaptive_max_pool3d]Trying to create tensor with negative dimension -1: [7, 5, -1, 3, 5]
 292. [torch._C._nn.avg_pool2d]pad should be at most half of effective kernel size, but got pad=4, kernel_size=-3 and dilation=1
 293. [torch._C._nn.avg_pool2d]pad must be non-negative, but got pad: -3
 294. [torch._C._nn.avg_pool2d]Dimension specified as -3 but tensor has no dimensions
 295. [torch._C._nn.avg_pool2d]divisor must be not zero
 296. [torch._C._nn.avg_pool2d]negative dimensions are not allowed
 297. [torch._C._nn.avg_pool2d]stride should not be zero
 298. [torch._C._nn.avg_pool2d]Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got:[1, 1, 1, 2, 1, 3, 1]
 299. [torch._C._nn.avg_pool2d]kernel size should be greater than zero, but got kH: 0 kW: 7
 300. [torch._C._nn.avg_pool2d]Expected out tensor to have dtype int, but got float instead
 301. [torch._C._nn.avg_pool2d]Given input size: (7x1x4). Calculated output size: (7x-2x-2). Output size is too small
 302. [torch._C._nn.avg_pool2d]Trying to resize storage that is not resizable
 303. [torch._C._nn.avg_pool3d]kernel size should be greater than zero, but got kT: 0 kH: 4 kW: 3
 304. [torch._C._nn.avg_pool3d]input image (T: 1 H: 4 W: 4) smaller than kernel size (kT: 4 kH: 5 kW: 2)
 305. [torch._C._nn.avg_pool3d]avg_pool3d(): Expected input's non-batch dimensions to have positive length, but input has a shape of [7, 0, 7, 0, 1] and non-batch dimension 0 has length zero!
 306. [torch._C._nn.avg_pool3d]divisor must be not zero
 307. [torch._C._nn.avg_pool3d]non-empty 4D or 5D (batch mode) tensor expected for input
 308. [torch._C._nn.avg_pool3d]negative dimensions are not allowed
 309. [torch._C._nn.avg_pool3d]pad should be at most half of effective kernel size, but got pad=7, kernel_size=5 and dilation=1
 310. [torch._C._nn.avg_pool3d]pad must be non-negative, but got pad: -2
 311. [torch._C._nn.gelu]GeluKernelImpl not implemented for 'Long'
 312. [torch._C._nn.huber_loss]huber_cpu not implemented for 'ComplexFloat'
 313. [torch._C._nn.log_sigmoid]expected scalar type Float but found Int
 314. [torch._C._nn.log_sigmoid]negative dimensions are not allowed
 315. [torch._C._nn.max_pool2d_with_indices]pad must be non-negative, but got pad: -3
 316. [torch._C._nn.max_pool2d_with_indices]pad should be at most half of effective kernel size, but got pad=0, kernel_size=-1 and dilation=6
 317. [torch._C._nn.max_pool2d_with_indices]pad should be smaller than or equal to half of kernel size, but got padW = 7, padH = 5, kW = 7, kH = 5
 318. [torch._C._nn.max_pool2d_with_indices]stride should not be zero
 319. [torch._C._nn.mse_loss]reduction == Reduction::Mean || reduction == Reduction::Sum INTERNAL ASSERT FAILED at ../aten/src/ATen/native/Loss.cpp:97, please report a bug to PyTorch. 
 320. [torch._C._nn.mse_loss]Trying to resize storage that is not resizable
 321. [torch._C._nn.mse_loss]result type Float can't be cast to the desired output type Int
 322. [torch._C._nn.reflection_pad1d]Dimension specified as 0 but tensor has no dimensions
 323. [torch._C._nn.smooth_l1_loss]reduction == Reduction::Mean || reduction == Reduction::Sum INTERNAL ASSERT FAILED at ../aten/src/ATen/native/Loss.cpp:86, please report a bug to PyTorch. 
 324. [torch._C._nn.smooth_l1_loss]smooth_l1_loss does not support negative values for beta.
 325. [torch._C._nn.smooth_l1_loss]Trying to resize storage that is not resizable
 326. [torch._C._nn.smooth_l1_loss]result type Float can't be cast to the desired output type Int
 327. [torch._C._nn.soft_margin_loss]Trying to resize storage that is not resizable
 328. [torch._C._nn.soft_margin_loss]result type Float can't be cast to the desired output type Int
 329. [torch._C._nn.softplus]softplus_cpu not implemented for 'Long'
 330. [torch._C._nn.softplus]softplus_cpu not implemented for 'Bool'
 331. [torch._C._nn.softshrink]softshrink_cpu not implemented for 'Long'
 332. [torch._C._nn.softshrink]softshrink_cpu not implemented for 'Int'
 333. [torch._C._nn.upsample_bicubic2d]Must specify exactly one of output_size and scale_factors
 334. [torch._C._nn.upsample_bicubic2d]Expected static_cast<int64_t>(scale_factors->size()) == spatial_dimensions to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 335. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'Bool'
 336. [torch._C._nn.upsample_bicubic2d]Trying to resize storage that is not resizable
 337. [torch._C._nn.upsample_bicubic2d]negative dimensions are not allowed
 338. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'Int'
 339. [torch._C._nn.upsample_bicubic2d]Non-empty 4D data tensor expected but got a tensor with sizes [6, 0, 1, 4]
 340. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'Long'
 341. [torch._C._nn.upsample_bicubic2d]compute_indices_weights_cubic not implemented for 'Short'
 342. [torch._C._nn.upsample_bilinear2d]Must specify exactly one of output_size and scale_factors
 343. [torch._C._nn.upsample_bilinear2d]It is expected output_size equals to 2, but got size 0
 344. [torch._C._nn.upsample_bilinear2d]ArrayRef: invalid index Index = 0; Length = 0
 345. [torch._C._nn.upsample_bilinear2d]upsample_bilinear2d_channels_last not implemented for 'Long'
 346. [torch._C._nn.upsample_bilinear2d]Trying to resize storage that is not resizable
 347. [torch._C._nn.upsample_bilinear2d]Expected out tensor to have dtype int, but got float instead
 348. [torch._C._nn.upsample_bilinear2d]Non-empty 4D data tensor expected but got a tensor with sizes [4, 0, 2, 7]
 349. [torch._C._nn.upsample_linear1d]Expected static_cast<int64_t>(output_size->size()) == spatial_dimensions to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 350. [torch._C._nn.upsample_linear1d]Must specify exactly one of output_size and scale_factors
 351. [torch._C._nn.upsample_linear1d]It is expected output_size equals to 1, but got size 4
 352. [torch._C._nn.upsample_linear1d]compute_indices_weights_linear not implemented for 'Long'
 353. [torch._C._nn.upsample_linear1d]compute_indices_weights_linear not implemented for 'Int'
 354. [torch._C._nn.upsample_nearest1d]Must specify exactly one of output_size and scale_factors
 355. [torch._C._nn.upsample_nearest1d]Input and output sizes should be greater than 0, but got input (W: 1) and output (W: 0)
 356. [torch._C._nn.upsample_nearest1d]It is expected output_size equals to 1, but got size 0
 357. [torch._C._nn.upsample_nearest1d]compute_indices_weights_nearest not implemented for 'Int'
 358. [torch._C._nn.upsample_nearest1d]Trying to resize storage that is not resizable
 359. [torch._C._nn.upsample_nearest1d]Expected out tensor to have dtype float, but got int instead
 360. [torch._C._nn.upsample_nearest1d]Non-empty 3D data tensor expected but got a tensor with sizes [4, 0, 4]
 361. [torch._C._nn.upsample_nearest2d]Expected static_cast<int64_t>(scale_factors->size()) == spatial_dimensions to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 362. [torch._C._nn.upsample_nearest2d]Must specify exactly one of output_size and scale_factors
 363. [torch._C._nn.upsample_nearest2d]It is expected output_size equals to 2, but got size 5
 364. [torch._C._nn.upsample_nearest2d]ArrayRef: invalid index Index = 0; Length = 0
 365. [torch._C._nn.upsample_nearest2d]Input and output sizes should be greater than 0, but got input (H: 5, W: 6) output (H: -2, W: 4)
 366. [torch._C._nn.upsample_nearest2d]Trying to resize storage that is not resizable
 367. [torch._C._nn.upsample_nearest2d]Expected out tensor to have dtype float, but got int instead
 368. [torch._C._nn.upsample_nearest2d]negative dimensions are not allowed
 369. [torch._C._nn.upsample_nearest3d]Must specify exactly one of output_size and scale_factors
 370. [torch._C._nn.upsample_nearest3d]Expected static_cast<int64_t>(scale_factors->size()) == spatial_dimensions to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 371. [torch._C._nn.upsample_nearest3d]It is expected output_size equals to 3, but got size 0
 372. [torch._C._nn.upsample_nearest3d]compute_indices_weights_nearest not implemented for 'Long'
 373. [torch._C._nn.upsample_nearest3d]Input and output sizes should be greater than 0, but got input (D: 7, H: 3, W: 3) output (D: -3, H: 7, W: 3)
 374. [torch._C._nn.upsample_nearest3d]Trying to resize storage that is not resizable
 375. [torch._C._nn.upsample_nearest3d]negative dimensions are not allowed
 376. [torch._C._nn.upsample_nearest3d]Expected out tensor to have dtype int, but got float instead
 377. [torch._C._nn.upsample_nearest3d]Non-empty 5D data tensor expected but got a tensor with sizes [7, 0, 6, 7, 6]
 378. [torch._C._nn.upsample_trilinear3d]Must specify exactly one of output_size and scale_factors
 379. [torch._C._nn.upsample_trilinear3d]Expected static_cast<int64_t>(output_size->size()) == spatial_dimensions to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 380. [torch._C._nn.upsample_trilinear3d]It is expected output_size equals to 3, but got size 0
 381. [torch._C._nn.upsample_trilinear3d]ArrayRef: invalid index Index = 0; Length = 0
 382. [torch._C._nn.upsample_trilinear3d]Non-empty 5D data tensor expected but got a tensor with sizes [7, 0, 7, 1, 6]
 383. [torch._C._nn.upsample_trilinear3d]Expected out tensor to have dtype float, but got int instead
 384. [torch._C._nn.upsample_trilinear3d]Input and output sizes should be greater than 0, but got input (D: 0, H: 4, W: 6) output (D: 7, H: 4, W: 2)
 385. [torch._C._special.special_chebyshev_polynomial_t]result type Float can't be cast to the desired output type Int
 386. [torch._C._special.special_chebyshev_polynomial_t]Trying to resize storage that is not resizable
 387. [torch._C._special.special_chebyshev_polynomial_u]negative dimensions are not allowed
 388. [torch._C._special.special_chebyshev_polynomial_u]The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 4
 389. [torch._C._special.special_chebyshev_polynomial_u]Trying to resize storage that is not resizable
 390. [torch._C._special.special_chebyshev_polynomial_u]result type Float can't be cast to the desired output type Int
 391. [torch._C._special.special_erfcx]negative dimensions are not allowed
 392. [torch._C._special.special_hermite_polynomial_he]The size of tensor a (2) must match the size of tensor b (7) at non-singleton dimension 5
 393. [torch._C._special.special_hermite_polynomial_he]result type Float can't be cast to the desired output type Int
 394. [torch._C._special.special_hermite_polynomial_he]negative dimensions are not allowed
 395. [torch._C._special.special_laguerre_polynomial_l]The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 6
 396. [torch._C._special.special_laguerre_polynomial_l]Trying to resize storage that is not resizable
 397. [torch._C._special.special_laguerre_polynomial_l]result type Float can't be cast to the desired output type Int
 398. [torch._C._special.special_ndtr]negative dimensions are not allowed
 399. [torch._C._special.special_scaled_modified_bessel_k0]scaled_modified_bessel_k0_cpu not implemented for 'Half'
 400. [torch._C._special.special_spherical_bessel_j0]negative dimensions are not allowed
 401. [torch._C._special.special_xlog1py]result type Float can't be cast to the desired output type Int
 402. [torch._C._special.special_xlog1py]negative dimensions are not allowed
 403. [torch._C._special.special_zeta]The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 5
 404. [torch._C._special.special_zeta]Trying to resize storage that is not resizable
 405. [torch._C._special.special_zeta]result type Float can't be cast to the desired output type Int
 406. [torch.adaptive_avg_pool1d]adaptive_avg_pool2d not implemented for 'Int'
 407. [torch.adaptive_avg_pool1d]adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes [7, 6, 1, 0] with dimension 3 being empty
 408. [torch.adaptive_avg_pool1d]mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Int
 409. [torch.addcdiv]Integer division with addcdiv is no longer supported, and in a future  release addcdiv will perform a true division of tensor1 and tensor2. The historic addcdiv behavior can be implemented as (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for integer inputs and as (input + value * tensor1 / tensor2) for float inputs. The future addcdiv behavior is just the latter implementation: (input + value * tensor1 / tensor2), for all dtypes.
 410. [torch.addcmul]negative dimensions are not allowed
 411. [torch.addmm]expand(torch.FloatTensor{[6, 3, 6, 6]}, size=[6, 3]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (4)
 412. [torch.addmm]self and mat2 must have the same dtype, but got Int and Float
 413. [torch.addmm]Trying to resize storage that is not resizable
 414. [torch.addmm]Expected out tensor to have dtype float, but got int instead
 415. [torch.addmv]Trying to resize storage that is not resizable
 416. [torch.addmv]Expected out tensor to have dtype float, but got int instead
 417. [torch.addmv]expected scalar type Int but found Float
 418. [torch.addr]expand(torch.FloatTensor{[7, 5, 6, 5, 1, 3, 3]}, size=[7, 7]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (7)
 419. [torch.addr]The expanded size of the tensor (5) must match the existing size (6) at non-singleton dimension 1.  Target sizes: [5, 5].  Tensor sizes: [1, 6]
 420. [torch.addr]Trying to resize storage that is not resizable
 421. [torch.all]Dimension out of range (expected to be in range of [-7, 6], but got 58)
 422. [torch.all]negative dimensions are not allowed
 423. [torch.all]all: You passed a dimname (string) to this op in place of a dimension index but it does not yet support this behavior. Please pass a dimension index to work around this.
 424. [torch.amax]amax(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
 425. [torch.amax]dim 0 appears multiple times in the list of dims
 426. [torch.amax]amax(): Expected reduction dim 2 to have non-zero size.
 427. [torch.amax]Trying to resize storage that is not resizable
 428. [torch.amin]amin(): Expected reduction dim -3 to have non-zero size.
 429. [torch.amin]dim 1 appears multiple times in the list of dims
 430. [torch.amin]amin(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
 431. [torch.amin]Trying to resize storage that is not resizable
 432. [torch.aminmax]aminmax: Expected reduction dim 4 to have non-zero size.
 433. [torch.any]negative dimensions are not allowed
 434. [torch.any]Dimension out of range (expected to be in range of [-7, 6], but got 79)
 435. [torch.any]Trying to resize storage that is not resizable
 436. [torch.any]any: You passed a dimname (string) to this op in place of a dimension index but it does not yet support this behavior. Please pass a dimension index to work around this.
 437. [torch.argmax]Expected out tensor to have dtype long int, but got float instead
 438. [torch.argmax]Dimension out of range (expected to be in range of [-7, 6], but got 79)
 439. [torch.argmax]argmax(): Expected reduction dim to be specified for input.numel() == 0.
 440. [torch.argmax]argmax(): Expected reduction dim 2 to have non-zero size.
 441. [torch.argmin]argmin(): Expected reduction dim 3 to have non-zero size.
 442. [torch.argsort]argsort: You passed a dimname (string) to this op in place of a dimension index but it does not yet support this behavior. Please pass a dimension index to work around this.
 443. [torch.as_strided]Tensor: invalid storage offset -4
 444. [torch.as_strided]mismatch in length of strides and shape
 445. [torch.as_strided]setStorage: sizes [], strides [], storage offset 0, and itemsize 4 requiring a storage size of 4 are out of bounds for storage of size 0
 446. [torch.asinh]negative dimensions are not allowed
 456. [torch.bernoulli]Expected p_in >= 0 && p_in <= 1 to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
 457. [torch.bernoulli]bernoulli_tensor_cpu_p_ not implemented for 'Int'
 458. [torch.bernoulli]Trying to resize storage that is not resizable
 459. [torch.binary_cross_entropy_with_logits]negative dimensions are not allowed
 460. [torch.bitwise_and]bitwise_and_cpu not implemented for 'Half'
 461. [torch.bitwise_left_shift]result type Float can't be cast to the desired output type Int
 462. [torch.bitwise_left_shift]negative dimensions are not allowed
 463. [torch.bitwise_not]bitwise_not_cpu not implemented for 'Half'
 464. [torch.bitwise_or]bitwise_or_cpu not implemented for 'Half'
 465. [torch.bitwise_right_shift]Trying to resize storage that is not resizable
 466. [torch.bitwise_right_shift]rshift_cpu not implemented for 'Float'
 467. [torch.bitwise_right_shift]result type Float can't be cast to the desired output type Int
 468. [torch.bitwise_xor]Trying to resize storage that is not resizabl
 470. [torch.block_diag]torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 0 has 7 dimensions
 471. [torch.bmm]Expected size for first two dimensions of batch2 tensor to be: [2, 6] but got: [5, 7].
 472. [torch.bmm]expected scalar type Float but found Int
 474. [torch.broadcast_tensors]The size of tensor a (4) must match the size of tensor b (0) at non-singleton dimension 
 476. [torch.broadcast_tensors]qint16
 477. [torch.cat]Sizes of tensors must match except in dimension 5. Expected size 5 but got size 7 for tensor number 2 in the list.
 478. [torch.cat]Tensors must have same number of dimensions: got 5 and 7
 479. [torch.cat]Name 'dkEx' not found in Tensor[None, None, None, None, None, None, None].
 480. [torch.cat]Name 'dim_index' not found in Tensor[].
 481. [torch.cat]zero-dimensional tensor (at position 0) cannot be concatenated
 482. [torch.cat]negative dimensions are not allowed
 483. [torch.cat]torch.cat(): input types can't be cast to the desired output type Int
 484. [torch.cholesky]cholesky: A must be batches of square matrices, but they are 2 by 4 matrices
 485. [torch.cholesky]cholesky: The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).
 486. [torch.cholesky]cholesky: The input tensor A must have at least 2 dimensions.
 487. [torch.cholesky]cholesky_cpu not implemented for 'Int'
 488. [torch.cholesky]cholesky: Expected result to be safely castable from Float dtype, but got result with dtype Int
 489. [torch.cholesky_inverse]cholesky_inverse_out_cpu not implemented for 'Int'
 490. [torch.cholesky_inverse]cholesky_inverse: Expected result to be safely castable from Float dtype, but got result with dtype Int
 491. [torch.clamp]torch.clamp: At least one of 'min' or 'max' must not be None
 492. [torch.clamp]The size of tensor a (6) must match the size of tensor b (2) at non-singleton dimension 6
 493. [torch.clamp]Trying to resize storage that is not resizable
 494. [torch.clamp]Found dtype Int but expected Float
 495. [torch.clamp]result type Float can't be cast to the desired output type Int
 496. [torch.clamp_]torch.clamp: At least one of 'min' or 'max' must not be None
 497. [torch.clamp_]output with shape [] doesn't match the broadcast shape [4, 6, 6, 2, 1, 5, 1]
 498. [torch.clamp_]result type Float can't be cast to the desired output type Int
 499. [torch.clamp_min_]output with shape [] doesn't match the broadcast shape [1, 1, 2, 1, 5, 1, 1]
 500. [torch.clamp_min_]result type Float can't be cast to the desired output type Int
 501. [torch.column_stack]Sizes of tensors must match except in dimension 1. Expected size 6 but got size 7 for tensor number 1 in the list.
 502. [torch.column_stack]Trying to resize storage that is not resizable
 503. [torch.column_stack]torch.cat(): input types can't be cast to the desired output type Int
 504. [torch.column_stack]Tensors must have same number of dimensions: got 5 and 6
 505. [torch.complex]Expected both inputs to be Half, Float or Double tensors but got ComplexFloat and Float
 506. [torch.corrcoef]corrcoef(): expected input to have two or fewer dimensions but got an input with 7 dimensions
 507. [torch.cos]negative dimensions are not allowed
 508. [torch.count_nonzero]negative dimensions are not allowed
 509. [torch.cov]cov(): expected fweights to have integral dtype but got fweights with Float dtype
 510. [torch.cov]cov(): expected aweights to have one or fewer dimensions but got aweights with 7 dimensions
 511. [torch.cov]cov(): fweights cannot be negative
 512. [torch.cummax]negative dimensions are not allowed
 513. [torch.cummax]Name 'RXcB' not found in Tensor[None, None, None, None, None, None, None].
 514. [torch.cummax]Name 'Jyql' not found in Tensor[].
 515. [torch.cummin]Name 'jwai' not found in Tensor[None, None, None, None, None, None, None].
 516. [torch.cummin]Name 'HyJF' not found in Tensor[].
 517. [torch.cumprod]Name 'HfWK' not found in Tensor[None, None, None, None, None, None, None].
 518. [torch.cumprod]Name 'iyxn' not found in Tensor[].
 519. [torch.cumprod]negative dimensions are not allowed
 520. [torch.cumsum]Name 'hWQR' not found in Tensor[None, None, None, None, None, None, None].
 521. [torch.cumsum]Name 'vTES' not found in Tensor[].
 522. [torch.cumsum]negative dimensions are not allowed
 523. [torch.cumulative_trapezoid]slice() cannot be applied to a 0-dim tensor.
 524. [torch.cumulative_trapezoid]negative dimensions are not allowed
 525. [torch.diagonal]Name 'oMMb' not found in Tensor[None, None, None, None, None, None, None].
 526. [torch.diagonal]Name 'rnNZ' not found in Tensor[].
 527. [torch.diagonal_copy]Expected out tensor to have dtype int, but got float instead
 528. [torch.diff]diff expects the shape of tensor to prepend or append to match that of input except along the differencing dimension; input.size(0) = 6, but got tensor.size(0) = 3
 529. [torch.diff]Dimension out of range (expected to be in range of [-7, 6], but got 20)
 530. [torch.diff]order must be non-negative but got -4
 531. [torch.diff]diff expects input to be at least one-dimensional
 532. [torch.diff]diff expects prepend or append to be the same dimension as input
 533. [torch.diff]Trying to resize storage that is not resizable
 534. [torch.div]The size of tensor a (6) must match the size of tensor b (2) at non-singleton dimension 6
 535. [torch.div]negative dimensions are not allowed
 536. [torch.divide]div expected rounding_mode to be one of None, 'trunc', or 'floor' but found 'JGqp'
 537. [torch.divide]The size of tensor a (6) must match the size of tensor b (7) at non-singleton dimension 5
 538. [torch.divide]Trying to resize storage that is not resizable
 539. [torch.dstack]Sizes of tensors must match except in dimension 2. Expected size 1 but got size 2 for tensor number 1 in the list.
 540. [torch.dstack]Tensors must have same number of dimensions: got 5 and 6
 541. [torch.dstack]Trying to resize storage that is not resizable
 542. [torch.dstack]torch.cat(): input types can't be cast to the desired output type Int
 543. [torch.flatten]flatten() has invalid args: start_dim cannot come after end_dim
 544. [torch.flatten]Name 'dRfP' not found in Tensor[None, None, None, None, None, None, None].
 545. [torch.flatten]Name 'RkcU' not found in Tensor[].
 546. [torch.flip]dim 4 appears multiple times in the list of dims
 547. [torch.float_power]The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 5
 548. [torch.float_power]negative dimensions are not allowed
 549. [torch.floor_divide]ZeroDivisionError
 550. [torch.floor_divide]result type Float can't be cast to the desired output type Int
 551. [torch.fmax]negative dimensions are not allowed
 552. [torch.fmax]result type Float can't be cast to the desired output type Int
 553. [torch.fmod]negative dimensions are not allowed
 554. [torch.fmod]ZeroDivisionError
 557. [torch.frac]frac_cpu not implemented for 'Bool'
 558. [torch.gcd]result type Float can't be cast to the desired output type Int
 559. [torch.geqrf]geqrf_cpu not implemented for 'Half'
 560. [torch.gradient]torch.gradient expected each dimension size to be at least edge_order+1
 561. [torch.gradient]Dimension specified as 0 but tensor has no dimensions
 562. [torch.gradient]dim 4 appears multiple times in the list of dims
 563. [torch.gradient]torch.gradient only supports edge_order=1 and edge_order=2.
 564. [torch.gradient]Dimension out of range (expected to be in range of [-7, 6], but got 73)
 565. [torch.gradient]torch.gradient expected spacing to be unspecified, a scalar or it's spacing and dim arguments to have the same length, but got a spacing argument of length 0 and a dim argument of length 1.
 566. [torch.gradient]torch.gradient expected spacing to be unspecified, a scalar, or a list of length equal to 'self.dim() = 7', since dim argument was not given, but got a list of length 0
 570. [torch.gt]negative dimensions are not allowed
 571. [torch.gt]Trying to resize storage that is not resizable
 573. [torch.hardshrink]hardshrink_cpu not implemented for 'Long'
 574. [torch.heaviside]Trying to resize storage that is not resizable
 575. [torch.heaviside]heaviside is not yet implemented for tensors with different dtypes.
 576. [torch.histc]histogram_cpu not implemented for 'Long'
 577. [torch.histc]torch.histc: range of [-inf, -inf] is not finite
 578. [torch.histc]negative dimensions are not allowed
 579. [torch.histogram]torch.histogram(): bins must be > 0, but got 0 for dimension 0
 580. [torch.histogram]torch.histogramdd: input tensor and bins tensors should have the same dtype, but got input with dtype float and bins for dimension 0 with dtype int
 581. [torch.histogram]torch.histogramdd: if weight tensor is provided, input tensor and weight tensor should have the same dtype, but got input(float), and weight(int)
 582. [torch.histogram]torch.histogramdd: for a 1-dimensional histogram range should have 2 elements, but got 0
 583. [torch.histogram]torch.histogramdd: if weight tensor is provided it should have the same shape as the input tensor excluding its innermost dimension, but got input with shape [1, 1] and weight with shape [1512]
 584. [torch.histogram]histogramdd not implemented for 'Int'
 585. [torch.histogramdd]torch.histogramdd: for a 6-dimensional histogram range should have 12 elements, but got 3
 586. [torch.histogramdd]torch.histogram(): bins must be > 0, but got 0 for dimension 0
 587. [torch.histogramdd]torch.histogramdd: if weight tensor is provided it should have the same shape as the input tensor excluding its innermost dimension, but got input with shape [4, 4] and weight with shape [6, 6, 6]
 588. [torch.histogramdd]negative dimensions are not allowed
 589. [torch.histogramdd]histogramdd not implemented for 'Int'
 590. [torch.histogramdd]number of steps must be non-negative
 591. [torch.histogramdd]torch.histogramdd: if weight tensor is provided, input tensor and weight tensor should have the same dtype, but got input(float), and weight(int)
 592. [torch.histogramdd]torch.histogramdd: bins tensor should have at least 1 element, but got 0 elements in the bins tensor for dimension 3
 593. [torch.histogramdd]Dimension specified as -1 but tensor has no dimensions
 594. [torch.histogramdd][enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 2940367562500 bytes. Error code 12 (Cannot allocate memory)
 595. [torch.histogramdd]histogramdd: The size of bins must be equal to the innermost dimension of the input.
 596. [torch.histogramdd]torch.histogramdd: input tensor should have at least 2 dimensions
 598. [torch.histogramdd]qint1
 600. [torch.histogramdd]torch.histogramdd: input tensor and bins tensors should have the same dtype, but got input with dtype float and bins for dimension 0 with dtype int
 601. [torch.histogramdd]torch.histogramdd: expected 1 sequences of bin edges for a 1-dimensional histogram but got 2
 602. [torch.histogramdd]histogramdd(): argument 'bins' failed to unpack the object at pos 2 with error type must be tuple of ints,but got Tensor
 603. [torch.hstack]Sizes of tensors must match except in dimension 1. Expected size 7 but got size 5 for tensor number 1 in the list.
 604. [torch.hstack]Tensors must have same number of dimensions: got 2 and 7
 605. [torch.hstack]negative dimensions are not allowed
 606. [torch.hstack]torch.cat(): input types can't be cast to the desired output type Int
 607. [torch.hypot]Trying to resize storage that is not resizable
 608. [torch.hypot]result type Float can't be cast to the desired output type Int
 609. [torch.igamma]igamma_cpu not implemented for 'Int'
 610. [torch.igamma]Trying to resize storage that is not resizable
 611. [torch.igamma]result type Float can't be cast to the desired output type Int
 612. [torch.igammac]Trying to resize storage that is not resizable
 613. [torch.igammac]result type Float can't be cast to the desired output type Int
 614. [torch.index_put]ntensor >= 3 INTERNAL ASSERT FAILED at ../aten/src/ATen/native/cpu/IndexKernelUtils.h:10, please report a bug to PyTorch. 
 615. [torch.index_put]strides() called on an undefined Tensor
 616. [torch.index_put]negative dimensions are not allowed
 617. [torch.index_put]masked_fill_ only supports boolean masks, but got mask with dtype nullptr (uninitialized)
 619. [torch.index_put]Index put requires the source and destination dtypes match, got Float for the destination and Int for the sourc
 621. [torch.index_put]qint16
 622. [torch.index_put]too many indices for tensor of dimension 0 (got 2)
 623. [torch.isin]negative dimensions are not allowed
 624. [torch.kron]result type Float can't be cast to the desired output type Int
 625. [torch.kthvalue]kthvalue(): Expected reduction dim 3 to have non-zero size.
 626. [torch.kthvalue]Name 'ZSPg' not found in Tensor[None, None, None, None, None, None, None].
 627. [torch.kthvalue]Name 'dHmL' not found in Tensor[].
 628. [torch.lcm]lcm_cpu not implemented for 'Half'
 629. [torch.le]negative dimensions are not allowed
 630. [torch.lerp]The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 6
 631. [torch.lerp]expected dtype float for `end` but got dtype int
 632. [torch.lerp]Trying to resize storage that is not resizable
 633. [torch.lerp]result type Float can't be cast to the desired output type Int
 634. [torch.lerp]negative dimensions are not allowed
 635. [torch.lerp]Found dtype Int but expected Float
 636. [torch.log_softmax]Name 'dTBA' not found in Tensor[None, None, None].
 637. [torch.log_softmax]Name 'bfpL' not found in Tensor[].
 638. [torch.log_softmax]negative dimensions are not allowed
 639. [torch.logaddexp]logaddexp_cpu not implemented for 'Int'
 640. [torch.logaddexp]logaddexp_cpu not implemented for 'Char'
 641. [torch.logaddexp]logaddexp_cpu not implemented for 'Long'
 642. [torch.logaddexp]negative dimensions are not allowed
 643. [torch.logaddexp2]logaddexp2_cpu not implemented for 'ComplexDouble'
 644. [torch.logaddexp2]logaddexp2_cpu not implemented for 'Long'
 646. [torch.logaddexp2]negative dimensions are not allowed
 649. [torch.logcumsumexp]negative dimensions are not allowed
 650. [torch.logdet]logdet: Expected a floating point or complex tensor as input. Got Int
 651. [torch.logical_not]negative dimensions are not allowed
 652. [torch.logit]negative dimensions are not allowed
 653. [torch.logsumexp]output with shape [] doesn't match the broadcast shape [1, 1, 1, 1, 1, 1, 1]
 654. [torch.logsumexp]Dimension out of range (expected to be in range of [-2, 1], but got 95)

## Bad Error Messages 
 1. [torch.Tensor.remainder_]ZeroDivisionError
 2. [torch.logcumsumexp]Name 'njxg' not found in Tensor[None, None].
 3. [torch.index_put]Could not run 'aten::random_.from' with arguments from the 'QuantizedCPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::random_.from' is only available for these backends: [CPU, CUDA, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
    CPU: registered at aten/src/ATen/RegisterCPU.cpp:31470 [kernel]
    CUDA: registered at aten/src/ATen/RegisterCUDA.cpp:44611 [kernel]
    Meta: registered at aten/src/ATen/RegisterMeta.cpp:26984 [kernel]
    BackendSelect: fallthrough registered at ../aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    Python: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:153 [backend fallback]
    FuncTorchDynamicLayerBackMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:497 [backend fallback]
    Functionalize: registered at aten/src/ATen/RegisterFunctionalization_0.cpp:22277 [kernel]
    Named: fallthrough registered at ../aten/src/ATen/core/NamedRegistrations.cpp:11 [kernel]
    Conjugate: registered at ../aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
    Negative: registered at ../aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
    ZeroTensor: registered at ../aten/src/ATen/ZeroTensorFallback.cpp:86 [backend fallback]
    ADInplaceOrView: registered at ../torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp:4942 [kernel]
    AutogradOther: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradCPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradCUDA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradHIP: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradXLA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMPS: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradIPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradXPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradHPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradVE: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradLazy: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMTIA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse1: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse2: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse3: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMeta: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradNestedTensor: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    Tracer: registered at ../torch/csrc/autograd/generated/TraceType_1.cpp:16033 [kernel]
    AutocastCPU: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:378 [backend fallback]
    AutocastCUDA: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:244 [backend fallback]
    FuncTorchBatched: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
    BatchedNestedTensor: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
    FuncTorchVmapMode: registered at ../aten/src/ATen/functorch/BatchRulesRandomness.cpp:367 [kernel]
    Batched: registered at ../aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
    VmapMode: registered at ../aten/src/ATen/VmapModeRegistrations.cpp:37 [kernel]
    FuncTorchGradWrapper: registered at ../aten/src/ATen/functorch/TensorWrapper.cpp:202 [backend fallback]
    PythonTLSSnapshot: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:161 [backend fallback]
    FuncTorchDynamicLayerFrontMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:493 [backend fallback]
    PreDispatch: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:165 [backend fallback]
    PythonDispatcher: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:157 [backend fallback]

 4. [torch.histogramdd]Could not run 'aten::random_.from' with arguments from the 'QuantizedCPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::random_.from' is only available for these backends: [CPU, CUDA, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].CPU: registered at aten/src/ATen/RegisterCPU.cpp:31470 [kernel]
    CUDA: registered at aten/src/ATen/RegisterCUDA.cpp:44611 [kernel]
    Meta: registered at aten/src/ATen/RegisterMeta.cpp:26984 [kernel]
    BackendSelect: fallthrough registered at ../aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    Python: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:153 [backend fallback]
    FuncTorchDynamicLayerBackMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:497 [backend fallback]
    Functionalize: registered at aten/src/ATen/RegisterFunctionalization_0.cpp:22277 [kernel]
    Named: fallthrough registered at ../aten/src/ATen/core/NamedRegistrations.cpp:11 [kernel]
    Conjugate: registered at ../aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
    Negative: registered at ../aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
    ZeroTensor: registered at ../aten/src/ATen/ZeroTensorFallback.cpp:86 [backend fallback]
    ADInplaceOrView: registered at ../torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp:4942 [kernel]
    AutogradOther: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradCPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradCUDA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradHIP: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradXLA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMPS: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradIPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradXPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradHPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradVE: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradLazy: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMTIA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse1: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse2: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse3: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMeta: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradNestedTensor: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    Tracer: registered at ../torch/csrc/autograd/generated/TraceType_1.cpp:16033 [kernel]
    AutocastCPU: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:378 [backend fallback]
    AutocastCUDA: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:244 [backend fallback]
    FuncTorchBatched: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
    BatchedNestedTensor: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
    FuncTorchVmapMode: registered at ../aten/src/ATen/functorch/BatchRulesRandomness.cpp:367 [kernel]
    Batched: registered at ../aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
    VmapMode: registered at ../aten/src/ATen/VmapModeRegistrations.cpp:37 [kernel]
    FuncTorchGradWrapper: registered at ../aten/src/ATen/functorch/TensorWrapper.cpp:202 [backend fallback]
    PythonTLSSnapshot: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:161 [backend fallback]
    FuncTorchDynamicLayerFrontMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:493 [backend fallback]
    PreDispatch: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:165 [backend fallback]
    PythonDispatcher: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:157 [backend fallback]
 5. [torch.broadcast_tensors]Could not run 'aten::random_.from' with arguments from the 'QuantizedCPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::random_.from' is only available for these backends: [CPU, CUDA, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
    CPU: registered at aten/src/ATen/RegisterCPU.cpp:31470 [kernel]
    CUDA: registered at aten/src/ATen/RegisterCUDA.cpp:44611 [kernel]
    Meta: registered at aten/src/ATen/RegisterMeta.cpp:26984 [kernel]
    BackendSelect: fallthrough registered at ../aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    Python: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:153 [backend fallback]
    FuncTorchDynamicLayerBackMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:497 [backend fallback]
    Functionalize: registered at aten/src/ATen/RegisterFunctionalization_0.cpp:22277 [kernel]
    Named: fallthrough registered at ../aten/src/ATen/core/NamedRegistrations.cpp:11 [kernel]
    Conjugate: registered at ../aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
    Negative: registered at ../aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
    ZeroTensor: registered at ../aten/src/ATen/ZeroTensorFallback.cpp:86 [backend fallback]
    ADInplaceOrView: registered at ../torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp:4942 [kernel]
    AutogradOther: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradCPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradCUDA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradHIP: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradXLA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMPS: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradIPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradXPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradHPU: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradVE: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradLazy: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMTIA: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse1: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse2: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradPrivateUse3: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradMeta: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    AutogradNestedTensor: registered at ../torch/csrc/autograd/generated/VariableType_2.cpp:19741 [autograd kernel]
    Tracer: registered at ../torch/csrc/autograd/generated/TraceType_1.cpp:16033 [kernel]
    AutocastCPU: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:378 [backend fallback]
    AutocastCUDA: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:244 [backend fallback]
    FuncTorchBatched: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
    BatchedNestedTensor: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
    FuncTorchVmapMode: registered at ../aten/src/ATen/functorch/BatchRulesRandomness.cpp:367 [kernel]
    Batched: registered at ../aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
    VmapMode: registered at ../aten/src/ATen/VmapModeRegistrations.cpp:37 [kernel]
    FuncTorchGradWrapper: registered at ../aten/src/ATen/functorch/TensorWrapper.cpp:202 [backend fallback]
    PythonTLSSnapshot: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:161 [backend fallback]
    FuncTorchDynamicLayerFrontMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:493 [backend fallback]
    PreDispatch: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:165 [backend fallback]
    PythonDispatcher: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:157 [backend fallback]