#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
module @jit_elementwise attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x512x768xf32>, %arg1: tensor<768xf32>) -> (tensor<1x512x768xf32> {jax.result_info = "result"}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x512x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x512x768xf32>, tensor<768xf32>) outs(%0 : tensor<1x512x768xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.maximumf %2, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<1x512x768xf32>
    return %1 : tensor<1x512x768xf32>
  }
}

