#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir { class Pass; }

namespace remora::passes {

/// Outline MoE expert FFN subgraphs into separate @expert_slot_N functions.
///
/// Pattern-matches stablehlo::DotGeneralOp where lhsBatchingDimensions
/// contains 0 and lhs.shape[0] == numExperts.  Groups the matched gate / up /
/// down triples by top-k slot and emits one func.func per slot that takes
/// per-expert (non-batched) inputs.  @main is rewritten to slice the batched
/// tensors and call the outlined functions, then concatenate the results.
std::unique_ptr<mlir::Pass> createExpertOutliningPass(int64_t numExperts = 8);

/// Register ExpertOutliningPass in the global MLIR pass registry so it is
/// usable with --pass-pipeline='builtin.module(moe-expert-outlining)'.
void registerExpertOutliningPass();

} // namespace remora::passes
