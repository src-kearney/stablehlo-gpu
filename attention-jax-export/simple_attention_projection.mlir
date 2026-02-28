#loc1 = loc("x")
#loc2 = loc("w")
module @jit_attention_projection attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x512x768xf32> loc("x"), %arg1: tensor<768x768xf32> loc("w")) -> (tensor<1x512x768xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0] : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32> loc(#loc8)
    return %0 : tensor<1x512x768xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("/Users/seanrck/github/stablehlo-gpu/explore/attention_export.py":6:11 to :27)
#loc4 = loc("/Users/seanrck/github/stablehlo-gpu/explore/attention_export.py":11:11 to :54)
#loc5 = loc("attention_projection"(#loc3))
#loc6 = loc("<module>"(#loc4))
#loc7 = loc(callsite(#loc5 at #loc6))
#loc8 = loc("jit(attention_projection)/dot_general"(#loc7))

