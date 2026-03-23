# ARM64 Calling Convention: invokePacked vs direct call

`_mlir_ciface_main` expects three arguments. ARM64 C ABI passes the first 8 arguments in registers x0-x7.

---

## What `invokePacked` does (wrong)

`invokePacked` casts the function to `void(*)(void**)` and calls it with `args.data()` as the single argument.

```
void *args[] = { &result, &x_desc, &bias_desc };
(*fn)(args.data());   // fn treated as void(*)(void**)
```

```
         caller stack
        ┌─────────────┐
        │  &result    │ args[0]
        │  &x_desc    │ args[1]
        │  &bias_desc │ args[2]
        └──────┬──────┘
               │ args.data() = pointer to this array
               │
               ▼
┌──────────────────────────────┐
│  x0  │  args.data()         │  ← pointer to the array
│  x1  │  (garbage)           │  ← _mlir_ciface_main reads this as x_desc ptr
│  x2  │  (garbage)           │  ← _mlir_ciface_main reads this as bias_desc ptr
│  x3  │  (garbage)           │
│  ... │  ...                 │
└──────────────────────────────┘

_mlir_ciface_main sees:
  arg0 = args.data()     → tries to store result struct here  → corrupts the array
  arg1 = garbage         → tries to load x_desc from here    → SEGFAULT
  arg2 = garbage         → tries to load bias_desc from here → SEGFAULT
```

---

## What the direct call does (correct)

```
auto *fn = reinterpret_cast<void (*)(void *, void *, void *)>(*sym);
fn(&result, &x_desc, &bias_desc);
```

```
        &result      &x_desc    &bias_desc
           │             │           │
           ▼             ▼           ▼
┌──────────────────────────────────────────┐
│  x0  │  &result                         │  ← _mlir_ciface_main stores output here
│  x1  │  &x_desc                         │  ← loads x memref descriptor from here
│  x2  │  &bias_desc                      │  ← loads bias memref descriptor from here
│  x3  │  (unused)                        │
│  ... │  ...                             │
└──────────────────────────────────────────┘

_mlir_ciface_main sees:
  arg0 = &result     → stores filled StridedMemRefType<float,3> here  ✓
  arg1 = &x_desc     → loads x_desc, reads x_data correctly           ✓
  arg2 = &bias_desc  → loads bias_desc, reads bias_data correctly      ✓
```

---

## Why `reinterpret_cast` is needed

`engine.lookup()` returns `llvm::Expected<void*>` — a raw address with no type.
The compiler needs a typed function pointer to know how many registers to fill and with what.

```
void *raw = *sym;          // just a number — compiler emits no argument setup
raw(&result, ...);         // illegal: can't call void*

auto *fn = reinterpret_cast<void (*)(void*, void*, void*)>(raw);
fn(&result, &x_desc, &bias_desc);
// compiler now knows: put &result in x0, &x_desc in x1, &bias_desc in x2
// then BL (branch-and-link) to fn
```
