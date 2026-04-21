// TODO: migrate to TableGen (.td) when the dialect stabilizes.
// Dialect, ops, and passes are all in one file while the design is in flux.
// Split into Transforms/ subdirectory when passes stabilize.

#include "UserDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace remora::user {

// ---------------------------------------------------------------------------
// Dialect registration
// ---------------------------------------------------------------------------

UserDialect::UserDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx,
                    mlir::TypeID::get<UserDialect>()) {
  addOperations<RecencyOp, EngagementScoreOp, CollaborativeOp,
                ExplicitInterestOp, ExploreOp, GoalOp, PreferenceOp>();
}

// ---------------------------------------------------------------------------
// Goal helpers
// ---------------------------------------------------------------------------

llvm::StringRef goalToString(Goal g) {
  switch (g) {
  case Goal::ExplicitOverRevealed: return "explicit_over_revealed";
  case Goal::RevealedOverExplicit: return "revealed_over_explicit";
  case Goal::Balanced:             return "balanced";
  }
  llvm_unreachable("unknown Goal");
}

std::optional<Goal> goalFromString(llvm::StringRef s) {
  if (s == "explicit_over_revealed") return Goal::ExplicitOverRevealed;
  if (s == "revealed_over_explicit") return Goal::RevealedOverExplicit;
  if (s == "balanced")               return Goal::Balanced;
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Signal op helpers
// ---------------------------------------------------------------------------

static mlir::Type featureTensorType(mlir::MLIRContext *ctx) {
  return mlir::RankedTensorType::get({32}, mlir::Float32Type::get(ctx));
}

static mlir::LogicalResult verifyWeight(mlir::Operation *op, float w) {
  if (w < -1.0f || w > 1.0f)
    return op->emitOpError() << "weight " << w << " out of range [-1.0, 1.0]";
  return mlir::success();
}

// ---------------------------------------------------------------------------
// Signal ops (recency, engagement_score, collaborative, explicit_interest)
// ---------------------------------------------------------------------------

#define IMPL_SIGNAL_OP(ClassName, OpName, DimStart, DimEnd)                    \
  llvm::StringRef ClassName::getOperationName() { return "user." #OpName; }   \
  void ClassName::build(mlir::OpBuilder &b, mlir::OperationState &s,          \
                        mlir::Value input, float weight) {                     \
    s.operands.push_back(input);                                              \
    s.addTypes(featureTensorType(b.getContext()));                             \
    s.addAttribute("weight", b.getF32FloatAttr(weight));                      \
  }                                                                            \
  float ClassName::getWeight() {                                               \
    return (*this)->getAttrOfType<mlir::FloatAttr>("weight")                  \
        .getValue().convertToFloat();                                          \
  }                                                                            \
  mlir::LogicalResult ClassName::verify() {                                   \
    return verifyWeight(*this, getWeight());                                   \
  }                                                                            \
  mlir::ParseResult ClassName::parse(mlir::OpAsmParser &p,                    \
                                     mlir::OperationState &r) {               \
    mlir::OpAsmParser::UnresolvedOperand operand;                             \
    mlir::Type type;                                                          \
    double weight;                                                            \
    if (p.parseOperand(operand) || p.parseKeyword("weight") ||               \
        p.parseColon() || p.parseFloat(weight) ||                            \
        p.parseColonType(type))                                               \
      return mlir::failure();                                                 \
    llvm::SmallVector<mlir::Value, 1> resolved;                              \
    if (p.resolveOperand(operand, type, resolved))                           \
      return mlir::failure();                                                 \
    r.operands.append(resolved.begin(), resolved.end());                     \
    r.addAttribute("weight",                                                 \
        p.getBuilder().getF32FloatAttr(static_cast<float>(weight)));         \
    r.addTypes(type);                                                        \
    return mlir::success();                                                  \
  }                                                                            \
  void ClassName::print(mlir::OpAsmPrinter &p) {                              \
    p << " " << getOperand() << " weight: " << getWeight()                    \
      << " : " << getOperand().getType();                                      \
  }

IMPL_SIGNAL_OP(RecencyOp,          recency,          0,  8)
IMPL_SIGNAL_OP(EngagementScoreOp,  engagement_score, 8,  16)
IMPL_SIGNAL_OP(CollaborativeOp,    collaborative,    16, 24)
IMPL_SIGNAL_OP(ExplicitInterestOp, explicit_interest,24, 32)

#undef IMPL_SIGNAL_OP

// ---------------------------------------------------------------------------
// ExploreOp
// ---------------------------------------------------------------------------

llvm::StringRef ExploreOp::getOperationName() { return "user.explore"; }

void ExploreOp::build(mlir::OpBuilder &b, mlir::OperationState &s, float value) {
  s.addAttribute("value", b.getF32FloatAttr(value));
}

float ExploreOp::getValue() {
  return (*this)->getAttrOfType<mlir::FloatAttr>("value")
      .getValue().convertToFloat();
}

mlir::LogicalResult ExploreOp::verify() {
  float v = getValue();
  if (v < 0.0f || v > 1.0f)
    return emitOpError() << "value " << v << " out of range [0.0, 1.0]";
  return mlir::success();
}

mlir::ParseResult ExploreOp::parse(mlir::OpAsmParser &p,
                                   mlir::OperationState &r) {
  double value;
  if (p.parseKeyword("value") || p.parseColon() || p.parseFloat(value))
    return mlir::failure();
  r.addAttribute("value",
      p.getBuilder().getF32FloatAttr(static_cast<float>(value)));
  return mlir::success();
}

void ExploreOp::print(mlir::OpAsmPrinter &p) {
  p << " value: " << getValue();
}

// ---------------------------------------------------------------------------
// GoalOp
// ---------------------------------------------------------------------------

llvm::StringRef GoalOp::getOperationName() { return "user.goal"; }

void GoalOp::build(mlir::OpBuilder &b, mlir::OperationState &s, Goal goal) {
  s.addAttribute("goal",
      b.getStringAttr(goalToString(goal)));
}

Goal GoalOp::getGoal() {
  auto str = (*this)->getAttrOfType<mlir::StringAttr>("goal").getValue();
  return *goalFromString(str);
}

mlir::LogicalResult GoalOp::verify() {
  auto str = (*this)->getAttrOfType<mlir::StringAttr>("goal").getValue();
  if (!goalFromString(str))
    return emitOpError() << "unknown goal '" << str << "'";
  return mlir::success();
}

mlir::ParseResult GoalOp::parse(mlir::OpAsmParser &p, mlir::OperationState &r) {
  llvm::StringRef keyword;
  if (p.parseKeyword(&keyword))
    return mlir::failure();
  r.addAttribute("goal", mlir::StringAttr::get(p.getContext(), keyword));
  return mlir::success();
}

void GoalOp::print(mlir::OpAsmPrinter &p) {
  p << " " << goalToString(getGoal());
}

// ---------------------------------------------------------------------------
// PreferenceOp
// ---------------------------------------------------------------------------

llvm::StringRef PreferenceOp::getOperationName() { return "user.preference"; }

void PreferenceOp::build(mlir::OpBuilder &b, mlir::OperationState &s,
                         llvm::StringRef text) {
  s.addAttribute("text", b.getStringAttr(text));
}

llvm::StringRef PreferenceOp::getText() {
  return (*this)->getAttrOfType<mlir::StringAttr>("text").getValue();
}

mlir::LogicalResult PreferenceOp::verify() { return mlir::success(); }

mlir::ParseResult PreferenceOp::parse(mlir::OpAsmParser &p,
                                      mlir::OperationState &r) {
  mlir::StringAttr textAttr;
  if (p.parseAttribute(textAttr))
    return mlir::failure();
  r.addAttribute("text", textAttr);
  return mlir::success();
}

void PreferenceOp::print(mlir::OpAsmPrinter &p) {
  p << " \"" << getText() << "\"";
}

// ---------------------------------------------------------------------------
// InterpretPreferencesPass
// ---------------------------------------------------------------------------

namespace {

static std::string toLower(llvm::StringRef s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

static bool contains(const std::string &haystack, llvm::StringRef needle) {
  return haystack.find(needle.str()) != std::string::npos;
}

struct InterpretPreferencesPass
    : public mlir::PassWrapper<InterpretPreferencesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterpretPreferencesPass)

  std::string preferenceText;
  InterpretPreferencesPass() = default;
  InterpretPreferencesPass(llvm::StringRef text) : preferenceText(text.str()) {}

  llvm::StringRef getArgument() const override {
    return "user-interpret-preferences";
  }

  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func.getContext());

    // Inject a synthetic user.preference if preferenceText was passed via flag.
    if (!preferenceText.empty()) {
      builder.setInsertionPointToStart(&func.getBody().front());
      builder.create<PreferenceOp>(func.getLoc(), preferenceText);
    }

    func->walk([&](PreferenceOp op) {
      std::string text = toLower(op.getText());
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();

      bool matched = false;
      bool negate = contains(text, "less");

      auto weight = [&](float w) { return negate ? -w : w; };

      if (contains(text, "recent") || contains(text, "new") ||
          contains(text, "latest")) {
        builder.create<RecencyOp>(loc, op->getBlock()->getParentOp()
            ->getRegion(0).front().getArgument(0), weight(0.4f));
        matched = true;
      }
      if (contains(text, "scroll") || contains(text, "binge") ||
          contains(text, "addictive") || contains(text, "engagement")) {
        builder.create<EngagementScoreOp>(loc, op->getBlock()->getParentOp()
            ->getRegion(0).front().getArgument(0), weight(-0.4f));
        matched = true;
      }
      if (contains(text, "actually enjoy") || contains(text, "genuinely like") ||
          contains(text, "really want")) {
        builder.create<ExplicitInterestOp>(loc, op->getBlock()->getParentOp()
            ->getRegion(0).front().getArgument(0), weight(0.4f));
        builder.create<EngagementScoreOp>(loc, op->getBlock()->getParentOp()
            ->getRegion(0).front().getArgument(0), weight(-0.3f));
        builder.create<GoalOp>(loc, Goal::ExplicitOverRevealed);
        matched = true;
      }
      if (contains(text, "discover") || contains(text, "explore") ||
          contains(text, "new things") || contains(text, "surprise")) {
        builder.create<ExploreOp>(loc, 0.7f);
        matched = true;
      }
      if (contains(text, "similar") || contains(text, "people like me") ||
          contains(text, "friends")) {
        builder.create<CollaborativeOp>(loc, op->getBlock()->getParentOp()
            ->getRegion(0).front().getArgument(0), weight(0.3f));
        matched = true;
      }

      if (!matched) {
        op.emitWarning()
            << "no preference keywords recognized, ignoring user.preference";
      } else {
        llvm::errs() << "Interpreted preference: \"" << op.getText() << "\"\n";
      }

      op.erase();
    });
  }
};

// ---------------------------------------------------------------------------
// UserToStablehloPass
// ---------------------------------------------------------------------------

// Feature tensor dimension ranges per signal.
struct SignalDims { int64_t start, end; };
static constexpr SignalDims kRecency          = {0,  8};
static constexpr SignalDims kEngagementScore  = {8,  16};
static constexpr SignalDims kCollaborative    = {16, 24};
static constexpr SignalDims kExplicitInterest = {24, 32};

struct UserToStablehloPass
    : public mlir::PassWrapper<UserToStablehloPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UserToStablehloPass)

  llvm::StringRef getArgument() const override { return "user-to-stablehlo"; }

  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func.getContext());

    // user.preference reaching this pass is a hard error.
    mlir::WalkResult prefCheck = func->walk([&](PreferenceOp op) {
      op.emitError(
          "user.preference op must be lowered before user-to-stablehlo");
      return mlir::WalkResult::interrupt();
    });
    if (prefCheck.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Apply goal weight adjustments before processing signal ops.
    func->walk([&](GoalOp op) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();
      mlir::Value input = op->getBlock()->getParentOp()
          ->getRegion(0).front().getArgument(0);
      switch (op.getGoal()) {
      case Goal::ExplicitOverRevealed:
        builder.create<ExplicitInterestOp>(loc, input,  0.4f);
        builder.create<EngagementScoreOp>(loc,  input, -0.3f);
        builder.create<CollaborativeOp>(loc,    input, -0.2f);
        break;
      case Goal::RevealedOverExplicit:
        builder.create<EngagementScoreOp>(loc,  input,  0.3f);
        builder.create<CollaborativeOp>(loc,    input,  0.2f);
        builder.create<ExplicitInterestOp>(loc, input, -0.4f);
        break;
      case Goal::Balanced:
        break;
      }
      op.erase();
    });

    // Lower user.explore — attach explore_value attr to the func op.
    func->walk([&](ExploreOp op) {
      func->setAttr("user.explore_value",
          mlir::FloatAttr::get(mlir::Float32Type::get(func.getContext()),
                               op.getValue()));
      op.erase();
    });

    // Lower signal ops to StableHLO slice/scale/update_slice chains.
    // Each op reads the result of the previous op; the final value replaces
    // the original feature tensor.
    // Note: full StableHLO lowering requires stablehlo headers — stubbed here
    // until the stablehlo dependency is confirmed available in this TU.
    // TODO: emit stablehlo.slice + stablehlo.multiply + stablehlo.dynamic_update_slice
    func->walk([&](mlir::Operation *op) {
      if (!llvm::isa<RecencyOp, EngagementScoreOp, CollaborativeOp,
                     ExplicitInterestOp>(op))
        return;
      // Placeholder: erase the op. Full lowering in next commit.
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    });
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass factories
// ---------------------------------------------------------------------------

std::unique_ptr<mlir::Pass> createInterpretPreferencesPass(
    llvm::StringRef preferenceText) {
  return std::make_unique<InterpretPreferencesPass>(preferenceText);
}

std::unique_ptr<mlir::Pass> createUserToStablehloPass() {
  return std::make_unique<UserToStablehloPass>();
}

} // namespace remora::user
