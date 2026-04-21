// TODO: migrate to TableGen (.td) when the dialect stabilizes.
// Current ops: recency, engagement_score, collaborative, explicit_interest,
// explore, goal, preference.

#pragma once
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"

namespace remora::user {

// ---------------------------------------------------------------------------
// Dialect
// ---------------------------------------------------------------------------

class UserDialect : public mlir::Dialect {
public:
  explicit UserDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "user"; }
};

// ---------------------------------------------------------------------------
// Goal enum
// ---------------------------------------------------------------------------

enum class Goal {
  ExplicitOverRevealed,
  RevealedOverExplicit,
  Balanced,
};

llvm::StringRef goalToString(Goal g);
std::optional<Goal> goalFromString(llvm::StringRef s);

// ---------------------------------------------------------------------------
// Signal ops — take tensor<32xf32>, return tensor<32xf32>
//
// weight: F32 in [-1.0, 1.0]
// final_scale = clamp(1.0 + weight, 0.0, 2.0)
// ---------------------------------------------------------------------------

class RecencyOp;
class EngagementScoreOp;
class CollaborativeOp;
class ExplicitInterestOp;

#define DECL_SIGNAL_OP(ClassName)                                              \
  class ClassName : public mlir::Op<ClassName,                                 \
                        mlir::OpTrait::OneOperand,                             \
                        mlir::OpTrait::OneResult> {                            \
  public:                                                                      \
    using Op::Op;                                                              \
    static llvm::StringRef getOperationName();                                 \
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {              \
      static llvm::StringRef names[] = {"weight"};                            \
      return names;                                                            \
    }                                                                          \
    static void build(mlir::OpBuilder &b, mlir::OperationState &s,            \
                      mlir::Value input, float weight);                        \
    float getWeight();                                                         \
    mlir::LogicalResult verify();                                              \
    static mlir::ParseResult parse(mlir::OpAsmParser &p,                      \
                                   mlir::OperationState &r);                  \
    void print(mlir::OpAsmPrinter &p);                                        \
  };

DECL_SIGNAL_OP(RecencyOp)
DECL_SIGNAL_OP(EngagementScoreOp)
DECL_SIGNAL_OP(CollaborativeOp)
DECL_SIGNAL_OP(ExplicitInterestOp)

#undef DECL_SIGNAL_OP

// ---------------------------------------------------------------------------
// Annotation ops — no tensor operand
// ---------------------------------------------------------------------------

// user.explore value: [0.0, 1.0] — scalar annotation, attaches to enclosing func
class ExploreOp : public mlir::Op<ExploreOp, mlir::OpTrait::ZeroOperands,
                                  mlir::OpTrait::ZeroResults> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName();
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
    static llvm::StringRef names[] = {"value"};
    return names;
  }
  static void build(mlir::OpBuilder &b, mlir::OperationState &s, float value);
  float getValue();
  mlir::LogicalResult verify();
  static mlir::ParseResult parse(mlir::OpAsmParser &p, mlir::OperationState &r);
  void print(mlir::OpAsmPrinter &p);
};

// user.goal — lowers to signal weight adjustments before user-to-stablehlo
class GoalOp : public mlir::Op<GoalOp, mlir::OpTrait::ZeroOperands,
                               mlir::OpTrait::ZeroResults> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName();
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
    static llvm::StringRef names[] = {"goal"};
    return names;
  }
  static void build(mlir::OpBuilder &b, mlir::OperationState &s, Goal goal);
  Goal getGoal();
  mlir::LogicalResult verify();
  static mlir::ParseResult parse(mlir::OpAsmParser &p, mlir::OperationState &r);
  void print(mlir::OpAsmPrinter &p);
};

// user.preference — free text, must be lowered by interpret-preferences before
// user-to-stablehlo runs. Fails with an error if it reaches user-to-stablehlo.
class PreferenceOp : public mlir::Op<PreferenceOp, mlir::OpTrait::ZeroOperands,
                                     mlir::OpTrait::ZeroResults> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName();
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
    static llvm::StringRef names[] = {"text"};
    return names;
  }
  static void build(mlir::OpBuilder &b, mlir::OperationState &s,
                    llvm::StringRef text);
  llvm::StringRef getText();
  mlir::LogicalResult verify();
  static mlir::ParseResult parse(mlir::OpAsmParser &p, mlir::OperationState &r);
  void print(mlir::OpAsmPrinter &p);
};

// ---------------------------------------------------------------------------
// Passes
// ---------------------------------------------------------------------------

// Lowers user.preference (free text) to typed user dialect ops.
// Run before user-to-stablehlo.
std::unique_ptr<mlir::Pass> createInterpretPreferencesPass(
    llvm::StringRef preferenceText = "");

// Lowers all user dialect ops to StableHLO.
// Run after interpret-preferences, before stablehlo-legalize-to-linalg.
std::unique_ptr<mlir::Pass> createUserToStablehloPass();

} // namespace remora::user
