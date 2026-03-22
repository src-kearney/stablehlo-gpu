## DialectRegistry

> What is it storing internally? (Look at mlir/include/mlir/IR/DialectRegistry.h)

A map of { dialect namespace -> constructor for matching dialect }

> What's the difference between registering a dialect and loading a dialect?

Loading a dialect refers to loading it into Context (eager - "build it now"), registering a dialect adds it to the {namespace -> constructor} `DialectRegistry` map (lazy - "I know how to build this when needed").

> Why does it exist separately from MLIRContext?

Decouples the list of dialects "available" from the dialects loaded in the Context. This is useful because the parser lazily loads dialects in the t as ops are encountered.

> What order do registerAllDialects, registerAllExtensions, and stablehlo::registerAllDialects need to go in and why?

All dialects before extensions. Extensions attach to op/type definitions that must already exist in the registry. Extension registration walks the registry to find dialects it extends in the namespace, so they must be registered first.

Otherwise, for dialect registration, upstream-before-downstream is conventional.

## MLIRContext

> What does it own? (Look at mlir/include/mlir/IR/MLIRContext.h, skim the private members)

MLIRContext handles loading of a dialect. It wraps some multi-threading capabilities and by default creates a thread pool (footgun if multiple contexts exist). The `DialectRegistry` is the factory, and once a dialect is loaded, the context holds it. Every `Type` and `Attribute` is interned per-context. `IntegerType::get(&ctx, 32)` returns a pointer into the Context's uniquer storage instead of allocating a new object every time.

> What is interning?

A memory optimization where you guarantee at most one copy of any logically equal value exist, and hand out pointers to that canonical copy.

```
Type t1 = IntegerType::get(&ctx, 32);
Type t2 = IntegerType::get(&ctx, 32);
assert(t1 == t2); // same pointer
```

The payoff is that equality checks become pointer comparisons (O(1)) instead of structural recursion. Downside being that interned objects are immutable and context-scoped - you can't modify a type after the fact since it's shared.

> Why must Context outlive every op created in it?

Context must outlive every op created in it because ops don't own any of their constituent data. They instead hold pointers into context-owned storage.

An op holds:
- Type pointers: into context's uniquer (interned table)
- Attribute pointers: An op's attribute dict is a set of `(StringAttr, Attribute)` pairs, both interned in context
- `AbstractOperation *`: shared struct holding function pointers for verify/print/parse (like MLIR's manual vtable) - without this, op's pointers are left dangling
- Identifier/string data: op names, attribute keys, backed by context-owned interned strings

These are raw pointers into allocations owned by Context. Context is the allocator, ops are borrowers.

## OwningOpRef<ModuleOp>

> What is OwningOpRef and what problem does it solve?

OwningOpRef does what it says - owns the reference to an op. Automatically destroys the held op on destruction.

OpBuilder is preferred over OwningOpRef.

> What does "owning" mean here vs a raw ModuleOp *?

With a raw ModuleOp *, nothing destroys the op when it goes out of scope. OwningOpRef<ModuleOp> is a RAII wrapper that calls erase() in its destructor.


## parseSourceFile

> What does the <mlir::ModuleOp> template argument do?

Specificies expected root op type. `parseSourceFile` is a template, the type tells it what to verify the top-level op is + return type after parsing.

> What does it return on failure — null, or does it throw?

Empty/null `OwningOpRef`. Check with `if (!module)`. MLIR strongly prefers LogicalResult / null returns over exceptions throughout.

> Where does the parsed IR live in memory — on the heap, in the context?

On the heap, allocated through Context's internal allocator. Context owns type/attribute storage. The ops themselves are heap-allocated and owned by whoever holds `OwningOpRef`. Destroying Context while ops are alive results in dangling pointers.

> Why does parseSourceFile take a pointer to ctx rather than a value?

1. `MLIRContext` is non-copyable - it owns unique, non-duplicable resources.
2. Everything parsed holds raw pointers into the specific Context instance.

> What happens if you add a pass that requires a dialect that isn't registered?

Pass manager fails with `LogicalResult::failure` in pass precondition verification before running.

> What does DialectRegistry do and why does it exist separately from MLIRContext?

> Why does the module->walk to add llvm.emit_c_interface have to run before pm.run and not after?

> What is _mlir_ciface_main and why does engine.lookup look for that name instead of just main?

> What does reconcile-unrealized-casts do and why is it the last pass?

> Why does the kernel do its own malloc internally, and why do you free(result.basePtr) instead of free(result.data)?

## Kernel invocation

> StridedMemRefType has two pointers: basePtr and data. What is each one for, and why do we set them to the same value for our input descriptors?

`basePtr` is the pointer to the original allocation — the one you pass to `free`. `data` is the pointer to where element [0,0,...,0] actually lives, which may be offset forward from `basePtr` for alignment reasons. When you do an aligned `malloc`, the allocator might hand you a slightly forward pointer for cache-line alignment; `basePtr` remembers where the actual allocation started so the runtime can free it correctly.

For our inputs we set both to `x_data.data()` because `std::vector` gives us a plain allocation with no alignment gap — offset is 0 and both pointers are the same.

> What does `reinterpret_cast<void (*)(void *, void *, void *)>(*sym)` actually do, and why is the cast necessary?

`engine.lookup()` returns the raw symbol address as a `void *` — just a number, no type information. The CPU doesn't know how to pass arguments to a function stored as `void *`. The `reinterpret_cast` tells the compiler "treat this address as a function that takes three `void *` arguments and returns nothing", so the compiler emits the correct calling-convention sequence: put `&result` in register x0, `&x_desc` in x1, `&bias_desc` in x2, then branch-and-link.

Without the cast, you can't call the pointer at all — C++ has no syntax for calling a `void *`.

> We originally tried `engine.invokePacked("_mlir_ciface_main", args)` and got a segfault. Why?

`invokePacked` treats the function as `void(*)(void**)` — it puts the address of the `args` array in a single register (x0) and calls. But `_mlir_ciface_main` was compiled expecting three separate `void*` arguments in x0, x1, x2. Those registers contain garbage, so it reads a bogus `x_desc` pointer and crashes. The fix is `engine.lookup` + a direct call with the correct 3-argument signature.

> Why does `createLinalgElementwiseOpFusionPass` need `createInlinerPass` to run before it?

The StableHLO file defines `@relu` as a separate private function and calls it from `@main`. After `createStablehloLegalizeToLinalgPass`, `@relu` is still a separate function — the fusion pass can only fuse ops it can see together, and it can't look across a function call boundary.

`createInlinerPass` copies `@relu`'s body into every call site, so `@main` now contains all the `linalg.generic` ops (broadcast, add, maximum) in sequence. Only then can `createLinalgElementwiseOpFusionPass` collapse them into a single fused `linalg.generic` with all three ops in its body.

> What does `bufferizeFunctionBoundaries = true` do, and what error do you see without it?

By default, one-shot bufferization only bufferizes ops *inside* functions — it leaves function arguments and return types as tensors. With `bufferizeFunctionBoundaries = false`, `@main` ends up returning a tensor, `bufferization.to_tensor` ops survive, and when `createConvertFuncToLLVMPass` runs it fails because LLVM dialect has no tensor type.

Setting `bufferizeFunctionBoundaries = true` tells bufferization to also rewrite function signatures: `tensor<1x512x768xf32>` arguments become `memref<...>` arguments, and the return value becomes a `memref` too. This is what allows the entire pipeline to reach pure LLVM dialect with no tensors remaining.
