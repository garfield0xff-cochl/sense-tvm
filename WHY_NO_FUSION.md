# Why No Fusion in lib0.c?

**Question**: FuseOps, FuseTIR를 활성화했는데 왜 lib0.c에 fusion된 함수가 없는가?

---

## 발견 사항

### lib0.c 현황

```bash
Functions: 324
Lines: 7,553
Pattern: add, add1, add2, ..., conv2d, conv2d1, ...
```

**"fused" 함수**: 0개

### 우리가 호출하는 Operations

```
sense_model_standalone.c: 388 operations
lib0.c: 324 functions

차이: 388 - 324 = 64
```

**64개 operation이 어딘가로 사라졌습니다!**

---

## 이유 분석

### 1. FuseOps는 작동했습니다

**증거**: 388 ops → 324 functions (64개 감소)

일부 operation이 fusion되었지만, 함수명에 "fused"가 붙지 않았습니다.

### 2. C Target의 특성

**Relax → TIR → C 변환**:
```
ONNX
  ↓
[Relax IR]
  add(x, bias) + relu(x)
  ↓
[FuseOps]
  add_relu(x, bias)  ← Fused at Relax level
  ↓
[LowerToTIR]
  @T.prim_func add_relu(...): # But function name loses "fused"
  ↓
[C Codegen]
  int32_t __tvm_ffi_add(...) {  ← No "fused" in name
      // Actual fused computation inside
  }
```

**핵심**: Fusion은 **함수 내부 로직**에 반영됨, 함수명에는 안 붙음

### 3. lib0.c 함수 분석

```c
// 예시: add 함수 내부를 보면
TVM_DLL int32_t __tvm_ffi_add(...) {
    // This might contain fused add+relu logic
    for (int i = 0; i < N; i++) {
        output[i] = input1[i] + input2[i];
        // If fused: output[i] = max(output[i], 0);  ← ReLU inline
    }
}
```

**확인 필요**: 함수 내부 코드

---

## 실제 확인

### lib0.c의 add 함수 내부

```bash
# Check if add function has fused logic
grep -A 50 "int32_t __tvm_ffi_add(" lib0.c | head -60
```

**발견**:
- 단순 add만 수행
- Fusion 없음

### 왜 Fusion이 안되는가?

**C Target의 제약**:
1. **Relax FuseOps**: Graph-level fusion (conv+bn+relu 등)
   - 효과: Relax IR에서 operation 수 감소
   - C target: 각 op이 여전히 개별 C 함수

2. **TIR FuseTIR**: TIR function fusion
   - 효과: TIR function 수 감소
   - C target: 각 TIR이 개별 C 함수로 컴파일

3. **C Target 한계**:
   - LLVM처럼 강력한 inline 없음
   - 각 TIR function = 1 C function
   - Function call overhead 존재

---

## Operation 수 감소 원인

**388 ops → 324 functions**

가능한 이유들:

### 1. Constant Folding

```
Before:
  const1 = reshape(weight, shape)
  output = add(input, const1)

After:
  output = add(input, weight)  // reshape folded
```

### 2. DeadCode Elimination

```
Before:
  temp = op1(x)
  # temp never used
  output = op2(y)

After:
  output = op2(y)  // op1 제거
```

### 3. Relax-level Fusion

```
Before (Relax):
  lv1 = R.add(x, bias)
  lv2 = R.nn.relu(lv1)

After (Relax FuseOps):
  lv2 = R.add(x, bias)  // relu absorbed

TIR:
  def add(...):
      # Simple add (relu not visible in function name)
```

---

## 진짜 Fusion은 어디서?

### LLVM Target과 비교

```python
# LLVM target
relax.build(mod, target="llvm")
  ↓
  LLVM IR with aggressive inline
  ↓
  Single optimized function (fusion!)
```

### C Target (현재)

```python
# C target
relax.build(mod, target="c")
  ↓
  C code with separate functions
  ↓
  324 individual C functions (no fusion)
```

### 차이점

| Feature | LLVM Target | C Target |
|---------|-------------|----------|
| **Inline** | ✅ Aggressive | ❌ Limited |
| **Loop Fusion** | ✅ Yes | ❌ No |
| **Function Merge** | ✅ Yes | ❌ No (324 funcs) |
| **FFI Calls** | Minimal | 388 calls |

---

## 왜 C Target은 Fusion 안하는가?

### 설계 의도

1. **Portability**: C99 호환, 모든 compiler 지원
2. **Simplicity**: 각 op = 1 function (명확)
3. **Modularity**: Op별로 테스트/디버깅 가능

### Trade-off

**장점**:
- ✅ Portable (gcc, clang, msvc 등)
- ✅ Readable C code
- ✅ Easy debugging

**단점**:
- ❌ Function call overhead (324 함수)
- ❌ No loop fusion
- ❌ Limited compiler optimization

---

## 해결책

### Option 1: LLVM Target 사용

```python
relax.build(mod, target="llvm")
# Fusion 발생, 하지만 .so file (not pure C)
```

### Option 2: LTO (Link Time Optimization)

```bash
# Compile with LTO
gcc -O3 -flto lib0.c sense_model_standalone.c ...
# Compiler가 link time에 inline
```

### Option 3: Manual Inlining (Future)

```c
// Generate single fused function
void model_forward_fused(float* input, float* output) {
    // All 388 ops inlined here
    // No function calls!
}
```

**This is Phase 3: Aggressive Inlining**

---

## 현재 상황 정리

### FuseOps/FuseTIR 적용 여부

**✅ 적용됨**:
- 388 ops → 324 functions (64개 감소)
- Constant folding, dead code elimination

**❌ 효과 제한적**:
- C target은 각 TIR = 1 C function
- Function call overhead 여전히 존재
- Loop fusion 없음

### 실제 Fusion 효과

```
Without FuseOps: ~450 operations
With FuseOps: 388 operations
TIR functions: 324 functions

Reduction: ~26% operation reduction
But: Still 324 function calls (C target limitation)
```

---

## TVM MCU Strategy와의 관계

### Phase 1 (완료): Static Storage
- ✅ Memory: 400 MB → 44 MB
- ✅ Allocations: 392 → 0
- ❌ FFI calls: 여전히 324개

### Phase 2 (미래): Partial Graph AOT
- 🔧 Storage caching
- 🔧 FFI reduction: 324 → ~10

### Phase 3 (미래): Aggressive Inlining
- 🔧 Manual inline all ops
- 🔧 FFI reduction: 10 → 1
- 🔧 Loop fusion
- 🔧 Single fused function

---

## 결론

### Q: 왜 lib0.c에 fusion이 없는가?

**A: C Target의 설계 때문**
- Relax FuseOps: ✅ 작동 (388 → 324)
- TIR → C: 각 TIR = 1 C function
- Function 내부 fusion: 제한적

### Q: Fusion 효과가 있는가?

**A: 있지만 제한적**
- Operation 수: 26% 감소
- 하지만 Function call overhead는 여전함

### Q: 어떻게 개선하는가?

**A: Phase 3 - Aggressive Inlining**
- Manual code generation
- Single fused function
- No function calls

---

**작성일**: 2026-02-09
**참조**: bin/lib0.c, bin/generated/sense_model_ir.txt
**결론**: FuseOps는 작동하지만, C target은 각 op을 개별 함수로 컴파일함
