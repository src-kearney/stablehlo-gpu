#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: remora-compiler <input.mlir>\n";
    return 1;
  }

  /// https://github.com/llvm/llvm-project/blob/f46a5153850c1303d687233d4adf699b01041da8/mlir/include/mlir/IR/DialectRegistry.h#L134
  /// maps a dialect namespace to a constructor for the
  /// matching dialect. This allows for decoupling the list of dialects
  /// "available" from the dialects loaded in the Context.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);

  /// https://github.com/llvm/llvm-project/blob/f46a5153850c1303d687233d4adf699b01041da8/mlir/include/mlir/IR/MLIRContext.h#L41
  /// MLIRContext is the top-level object for a collection of MLIR operations. It
  /// holds immortal uniqued objects like types, and the tables used to unique
  /// them.
  /// The context wrap some multi-threading facilities, and in particular by
  /// default it will implicitly create a thread pool.
  mlir::MLIRContext ctx(registry);

  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());

  /// https://github.com/llvm/llvm-project/blob/f46a5153850c1303d687233d4adf699b01041da8/mlir/include/mlir/IR/OwningOpRef.h#L29
  /// This class acts as an owning reference to an op, and will automatically
  /// destroy the held op on destruction if the held op is valid.
  /// Note that OpBuilder and related functionality should be highly preferred
  /// instead, and this should only be used in situations where existing solutions
  /// are not viable.
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);
  if (!module) {
    llvm::errs() << "Failed to parse: " << argv[1] << "\n";
    return 1;
  }

  module->print(llvm::outs());
  return 0;
}
