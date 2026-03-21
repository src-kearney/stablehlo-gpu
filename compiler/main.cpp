#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
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
#include "llvm/Support/TargetSelect.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <vector>

// Run the elementwise kernel via the JIT engine and print a sample of the output.
// Inputs: x = all 1.0, bias = all -0.5
// Expected: relu(1.0 + (-0.5)) = relu(0.5) = 0.5 everywhere
static int runElementwise(mlir::ExecutionEngine &engine) {
  const int64_t N = 1, T = 512, D = 768;
  std::vector<float> x_data(N * T * D, 1.0f);
  std::vector<float> bias_data(D, -0.5f);

  StridedMemRefType<float, 3> x_desc;
  x_desc.basePtr = x_desc.data = x_data.data();
  x_desc.offset = 0;
  x_desc.sizes[0] = N; x_desc.sizes[1] = T; x_desc.sizes[2] = D;
  x_desc.strides[0] = T * D; x_desc.strides[1] = D; x_desc.strides[2] = 1;

  StridedMemRefType<float, 1> bias_desc;
  bias_desc.basePtr = bias_desc.data = bias_data.data();
  bias_desc.offset = 0;
  bias_desc.sizes[0] = D;
  bias_desc.strides[0] = 1;

  // result is malloc'd inside the kernel; the descriptor is filled by the wrapper
  StridedMemRefType<float, 3> result;

  // _mlir_ciface_main(result*, x*, bias*) — each arg is a pointer to a memref descriptor.
  // Call directly (not via invokePacked) to preserve the 3-arg C calling convention.
  auto sym = engine.lookup("_mlir_ciface_main");
  if (!sym) {
    llvm::handleAllErrors(sym.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  auto *fn = reinterpret_cast<void (*)(void *, void *, void *)>(*sym);
  fn(&result, &x_desc, &bias_desc);

  llvm::outs() << "result[0][0][0] = " << result.data[0] << " (expected 0.5)\n";
  llvm::outs() << "result[0][0][1] = " << result.data[1] << " (expected 0.5)\n";
  free(result.basePtr);
  return 0;
}

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
  mlir::registerAllExtensions(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  /// https://github.com/llvm/llvm-project/blob/f46a5153850c1303d687233d4adf699b01041da8/mlir/include/mlir/IR/MLIRContext.h#L41
  /// MLIRContext is the top-level object for a collection of MLIR operations. It
  /// holds immortal uniqued objects like types, and the tables used to unique
  /// them.
  /// The context wrap some multi-threading facilities, and in particular by
  /// default it will implicitly create a thread pool.
  mlir::MLIRContext ctx(registry);

  bool debug = argc > 2 && llvm::StringRef(argv[2]) == "--mlir-print-ir-after-all";
  if (debug) ctx.disableMultithreading(); // required for `enableIRPrinting`

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

  mlir::PassManager pm(&ctx);
  if (debug) pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/nullptr,
    /*shouldPrintAfterPass=*/[](mlir::Pass *, mlir::Operation *) { return true; },
    /*printModuleScope=*/true,
    /*printAfterOnlyOnChange=*/true
  );
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());

  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;

  /// lower to LLVM dialect
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts)); // tensor → memref (incl. return types)
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass()); // linalg → scf
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Must run before pass pipeline so createConvertFuncToLLVMPass generates the wrapper
  module->walk([](mlir::func::FuncOp func) {
    if (func.isPublic())
      func->setAttr("llvm.emit_c_interface",
        mlir::UnitAttr::get(func.getContext()));
  });
  
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return 1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto maybeEngine = mlir::ExecutionEngine::create(*module);
  if (!maybeEngine) {
    llvm::handleAllErrors(maybeEngine.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Failed to create ExecutionEngine: " << e.message() << "\n";
    });
    return 1;
  }

  return runElementwise(*maybeEngine.get());
}
