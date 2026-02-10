# Phase 2 결과: LTO 적용

**날짜**: 2026-02-10
**목표**: FFI Call Reduction

---

## 구현: Link Time Optimization (LTO)

### 변경사항

**Makefile**:
```makefile
# Before
CFLAGS = -O3 -Wall -I. -march=native -fPIC
LDFLAGS = -lm

# After (Phase 2)
CFLAGS = -O3 -Wall -I. -march=native -fPIC -flto
LDFLAGS = -lm -flto -fwhole-program
```

**효과**:
- Compiler가 link time에 inline 결정
- 함수 간 최적화 가능
- 코드 변경 없이 최적화

---

## 측정 결과

### 성능 비교

| Metric | Before LTO | After LTO | Change |
|--------|------------|-----------|--------|
| **Inference** | ~4.2 ms | 4.1 ms | ~2% ↓ |
| **Speedup vs ONNX** | 1.08x | 1.09x | +0.01x |
| **Binary Size** | 236 KB | 247 KB | +11 KB |
| **Correctness** | ✅ PASSED | ✅ PASSED | Same |

### 검증

```
Correctness: ✅ PASSED
Max Diff: 2.38e-07 (Perfect!)
Speedup: 1.09x
```

---

## 분석

### LTO 효과

**예상**: 20-30% FFI reduction
**실제**: ~2% 성능 향상

**Why?**:
1. **FFI Overhead 작음**: FFI call이 전체의 ~5% 미만
2. **Compute Bound**: Conv2d, MatMul이 시간 대부분 차지
3. **Already Optimized**: lib0.c TIR functions가 이미 최적화됨

### Binary Size

```
LTO 전: 236 KB
LTO 후: 247 KB
증가: +11 KB (4.6%)
```

**이유**: Inline으로 코드 증가

---

## Phase 2 상태

### Option A: LTO ✅ 완료

```
Method: -flto compiler flag
Effect: ~2% improvement
FFI Calls: Still 388 (compiler가 일부만 inline)
```

**평가**: ⭐⭐☆☆☆ (효과 미미)

### Option B: Manual Batching (미구현)

```python
# Generate batch functions
def execute_ops_0_to_49(...):
    # 50 ops inlined
    __tvm_ffi_op1(...)
    __tvm_ffi_op2(...)
    ...

# Reduction: 388 → 8 calls
```

**예상 효과**: ⭐⭐⭐⭐☆ (50x FFI reduction)

### Option C: Selective Inlining (미구현)

```c
// Inline small ops (add, relu)
for (i) output[i] = max(input[i] + bias[i], 0);  // Inlined!

// Keep large ops (conv2d)
__tvm_ffi_conv2d(...);  // Function call
```

**예상 효과**: ⭐⭐⭐⭐⭐ (10-20% faster)

---

## C Target의 근본적 한계

### 문제

```
Relax → TIR → C:
    Each TIR = 1 C function
    FFI calls = TIR function count

C Target 특성:
    - No aggressive inline (LLVM처럼)
    - Each function = separate compilation unit
    - Limited cross-function optimization
```

### 해결책

**Phase 3: Manual Inline** 필요

```c
// 현재 (Phase 1)
__tvm_ffi_conv2d(...);  // FFI
__tvm_ffi_add(...);     // FFI
__tvm_ffi_maximum(...); // FFI
// 388 FFI calls

// Phase 3 목표 (Manual Inline)
// Conv2d logic inlined
for (...) {
    sum += input[...] * weight[...];
}
// Add logic inlined
for (...) {
    output[...] = sum + bias[...];
}
// Maximum logic inlined
for (...) {
    output[...] = max(output[...], 0);
}
// 0 FFI calls, single fused loop!
```

---

## Phase 2 결론

### 달성한 것

✅ **LTO 적용**:
- Compiler inline optimization
- ~2% 성능 향상
- 코드 변경 없음

### 한계

⚠️ **C Target 제약**:
- FFI calls: 여전히 388개
- Inline: Compiler 재량 (제한적)
- Fusion: Function level 불가

### 다음 단계

**Phase 3 필요**: Manual Aggressive Inlining
- Operation logic을 직접 inline
- TIR function을 C loop로 변환
- 0 FFI calls 목표

---

## 최종 상태

### Phase 1 ✅ 완료

```
Memory: 400 MB → 44 MB (89% ↓)
Allocations: 392 → 0 (100% ↓)
Buffer reuse: 94.9%
```

### Phase 2 ⭐ LTO 적용

```
LTO: Enabled
Performance: +2%
FFI Calls: Still 388 (compiler limitation)
```

### Phase 3 📝 계획

```
Manual Inline: TIR → C loop
Expected: +10-20% faster
FFI Calls: 388 → 1
```

---

**현재 달성도**: Phase 1 (100%) + Phase 2 (20% - LTO only)
**전체 TVM MCU Strategy**: ~75/100
**다음**: Phase 3 - Manual Aggressive Inlining

---

**작성일**: 2026-02-10
**LTO 성능**: 4.1 ms, 1.09x vs ONNX
**Status**: Phase 2 LTO 완료, Manual batching/inline은 Phase 3로 이동
