#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module @jit_attention_projection attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x512x768xf32>, %arg1: tensor<768x768xf32>) -> (tensor<1x512x768xf32> {jax.result_info = "result"}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x512x768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x512x768xf32>, tensor<768x768xf32>) outs(%1 : tensor<1x512x768xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<1x512x768xf32>
    return %2 : tensor<1x512x768xf32>
  }
}

