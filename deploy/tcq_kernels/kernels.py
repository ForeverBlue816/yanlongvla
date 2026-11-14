"""CUDA kernels for the trellis-coded 2-bit format (bitshift trellis L = 16, K = 2, V = 2, 16 x 16 tiles, 512 x 2 table per layer,
row scales, two-sided randomized block-Walsh rotation) with 8-bit activations taken in the rotated space.

One quantized Linear y = W x (+ b), W = S_v H_m diag(r) Q H_n S_u (Q = decoded table values, r = row scale):
  pre    x (BF16 / FP32) -> sign flip -> block Walsh (FP32, in shared memory) -> per-token absmax -> INT8 q, scale s = absmax / 127
  gemm   acc = q Q8^T (INT32, exact), Q8 = INT8 table values; prefill: cuBLASLt (torch._int_mm) on decoded INT8 weights; expert: k_gemm_*
  post   z = float(acc) * s * rs  ->  block Walsh  ->  * sign  ->  + bias  ->  BF16
Arithmetic is fixed so that the PyTorch reference (tcq_format.reference_linear) is reproduced bit for bit: FP32 butterflies in the
order h = 1, 2, 4, ... (radix-4 passes here compute the same sums), scale 1/sqrt(w) as an FP32 multiply, FP32 division for the
activation quantizer, round-half-even, no FMA contraction (compiled with --fmad=false).
"""
import torch

from cuda_rt import compile_module

COMMON = r'''
typedef unsigned short u16; typedef unsigned int u32; typedef signed char i8; typedef unsigned long long u64;

__device__ __forceinline__ float bf2f(u16 b) { return __uint_as_float(((u32)b) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {          // round to nearest even, as c10::BFloat16
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ float ldx(const u16* p, size_t i) { return bf2f(p[i]); }
__device__ __forceinline__ float ldx(const float* p, size_t i) { return p[i]; }

// GELU (tanh approximation) on a BF16 value, as torch.nn.functional.gelu(x, approximate='tanh') computes it on CUDA: FP32 arithmetic,
// result rounded to BF16. GELU_MODE 1: plain; 2: x + kappa * x^3 contracted into one FMA; 0: table lookup. The mode is chosen at load
// by comparing all 65536 inputs with PyTorch's result on this device (kernels.pick_gelu), so the kernels match it exactly.
__device__ __forceinline__ u16 gelu_formula(u16 xb, int fma) {
    const float x = bf2f(xb);
    const float kBeta = (float)(1.41421356237309504880 * 1.12837916709551257390 * 0.5), kKappa = 0.044715f;
    const float x3 = x * x * x;
    const float inner = kBeta * (fma ? fmaf(kKappa, x3, x) : (x + kKappa * x3));
    return f2bf(0.5f * x * (1.f + tanhf(inner)));
}
__device__ __forceinline__ u16 gelu_bf16(u16 xb, const u16* lut) {
#if GELU_MODE == 0
    return lut[xb];
#else
    return gelu_formula(xb, GELU_MODE == 2);
#endif
}

// Walsh-Hadamard butterflies on a power-of-two segment of shared memory, all threads of the block cooperating. Stage order
// h = 1, 2, 4, ...; a radix-2^R pass performs R consecutive stages on 2^R values held in registers (the same additions, in the same
// order, as R radix-2 passes), so a 16384-wide transform is 4 sweeps of shared memory instead of 14. The last pass hands each
// finished value to f(index, value) (scale / store / quantizer statistics fused into the sweep).
template <int R>
__device__ __forceinline__ void bfly(float* v) {
    #pragma unroll
    for (int s = 0; s < R; ++s) {
        const int hh = 1 << s;
        #pragma unroll
        for (int k = 0; k < (1 << R); ++k)
            if (!(k & hh)) { const float a = v[k], b = v[k + hh]; v[k] = a + b; v[k + hh] = a - b; }
    }
}

template <int R>
__device__ __forceinline__ void pass_mid(float* buf, int w, int S) {
    const int T = blockDim.x, G = w >> R;
    for (int g = threadIdx.x; g < G; g += T) {
        const int low = g & (S - 1); const int i = ((g - low) << R) | low;
        float v[1 << R];
        #pragma unroll
        for (int k = 0; k < (1 << R); ++k) v[k] = buf[i + k * S];
        bfly<R>(v);
        #pragma unroll
        for (int k = 0; k < (1 << R); ++k) buf[i + k * S] = v[k];
    }
}

template <int R, class Fn>
__device__ __forceinline__ void pass_last(float* buf, int w, int S, Fn&& f) {
    const int T = blockDim.x, G = w >> R;
    for (int g = threadIdx.x; g < G; g += T) {
        const int low = g & (S - 1); const int i = ((g - low) << R) | low;
        float v[1 << R];
        #pragma unroll
        for (int k = 0; k < (1 << R); ++k) v[k] = buf[i + k * S];
        bfly<R>(v);
        #pragma unroll
        for (int k = 0; k < (1 << R); ++k) f(i + k * S, v[k]);
    }
}

// First pass (stages h = 1, 2, 4, 8 on 16 contiguous values) fused with the load: values come from ld(index) (global memory or the
// buffer itself) and go to shared memory. Thread g keeps element e of its group in register slot e ^ rot, rot = (g >> 1) & 15, so
// that the 32 threads of a warp hit 32 different shared-memory banks at every instruction (contiguous groups would otherwise put
// 16 threads on one bank). XOR keeps the butterfly pairing (slot j <-> slot j ^ h); when bit h of rot is set the two slots swap
// roles, so the sum goes to the upper slot and the negated difference to the lower one: the same additions, bit for bit.
template <class Ld>
__device__ __forceinline__ void pass_first(float* buf, int w, Ld&& ld) {
    const int T = blockDim.x, G = w >> 4;
    for (int g = threadIdx.x; g < G; g += T) {
        const int rot = (g >> 1) & 15, i = g << 4;
        float v[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) v[k] = ld(i + (k ^ rot));
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int hh = 1 << s; const bool sw = (rot & hh) != 0;
            #pragma unroll
            for (int k = 0; k < 16; ++k)
                if (!(k & hh)) { const float a = v[k], b = v[k + hh]; const float sum = a + b, d = a - b; v[k] = sw ? -d : sum; v[k + hh] = sw ? sum : d; }
        }
        #pragma unroll
        for (int k = 0; k < 16; ++k) buf[i + (k ^ rot)] = v[k];
    }
}

// Full transform of one power-of-two segment: ld(i) supplies the inputs (i in [0, w)), f(i, value) receives the outputs.
// rad = 4: radix-16 passes (few sweeps; best when many tokens keep every SM busy); rad = 2: radix-4 passes (more threads have work;
// best when there are few tokens). Both perform the same additions in the same order.
template <class Ld, class Fn>
__device__ __forceinline__ void walsh(float* buf, int w, int rad, Ld&& ld, Fn&& f) {
    int L = 31 - __clz(w), S = 1;
    for (int i = threadIdx.x; i < w; i += blockDim.x) buf[i] = ld(i);
    __syncthreads();
    if (rad == 4) {
        if (L > 4) { pass_first(buf, w, [&](int i) { return buf[i]; }); __syncthreads(); S = 16; L -= 4; }
        while (L > 4) { pass_mid<4>(buf, w, S); __syncthreads(); S <<= 4; L -= 4; }
    } else {
        while (L > 2) { pass_mid<2>(buf, w, S); __syncthreads(); S <<= 2; L -= 2; }
    }
    if (L == 4) pass_last<4>(buf, w, S, f);
    else if (L == 3) pass_last<3>(buf, w, S, f);
    else if (L == 2) pass_last<2>(buf, w, S, f);
    else if (L == 1) pass_last<1>(buf, w, S, f);
    else if (threadIdx.x == 0) f(0, buf[0]);
    __syncthreads();
}

// Block Walsh over the binary decomposition of K (descending widths, e.g. 4304 = 4096 + 128 + 64 + 16): ld(index) -> inputs,
// f(index, value * 1/sqrt(w)) <- outputs.
template <class Ld, class Fn>
__device__ __forceinline__ void block_walsh(float* buf, int K, int rad, Ld&& ld, Fn&& f) {
    int off = 0, rem = K;
    while (rem) {
        const int w = 1 << (31 - __clz(rem));
        const float c = (float)(1.0 / sqrt((double)w));
        float* b = buf + off; const int o = off;
        walsh(b, w, rad, [&](int i) { return ld(o + i); }, [&](int i, float v) { f(o + i, v * c); });
        off += w; rem -= w;
    }
}

// max over the block of each thread's running maximum (red: 32 floats of shared memory)
__device__ __forceinline__ float block_max(float mx, float* red) {
    const int T = blockDim.x, t = threadIdx.x;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((t & 31) == 0) red[t >> 5] = mx;
    __syncthreads();
    mx = 0.f;
    const int warps = (T + 31) >> 5;
    for (int i = 0; i < warps; ++i) mx = fmaxf(mx, red[i]);
    return mx;
}

__device__ __forceinline__ int q8(float v, float s) {
    float r = rintf(v / s);
    r = fminf(fmaxf(r, -127.f), 127.f);
    return (int)r;
}

// Per-token INT8 quantizer of a rotated vector in shared memory: scale = absmax / 127; q (M, K) INT8 row-major when q != 0, q4 (K/4, ldq)
// INT32 words (4 consecutive inputs per word, token = column) when q4 != 0.
__device__ __forceinline__ void quantize_store(const float* buf, float mx, float* red, i8* q, int* q4, float* scale, int tok, int K, int ldq) {
    const int T = blockDim.x, t = threadIdx.x;
    mx = block_max(mx, red);
    const float s = mx > 0.f ? mx / 127.f : 1.f;
    if (t == 0) scale[tok] = s;
    const size_t base = (size_t)tok * K;
    for (int j = t; j < (K >> 2); j += T) {
        const u32 word = (u32)(q8(buf[4 * j], s) & 0xff) | ((u32)(q8(buf[4 * j + 1], s) & 0xff) << 8)
                       | ((u32)(q8(buf[4 * j + 2], s) & 0xff) << 16) | ((u32)(q8(buf[4 * j + 3], s) & 0xff) << 24);
        if (q) ((u32*)(q + base))[j] = word;
        if (q4) q4[(size_t)j * ldq + tok] = (int)word;
    }
}

// State of weight pair p (0..7) of row r (0..15) of one tile = bits [32 r + 4 p, 32 r + 4 p + 16) of the tile's 512-bit stream
// (32 uint16 words, most significant bit first, tail-biting). Decodes the 16 INT8 weights of one tile row into 4 little-endian words.
__device__ __forceinline__ void decode_row(const u16* tile, int r, const i8* t8, int* w) {
    const u64 win = ((u64)tile[2 * r] << 32) | ((u64)tile[2 * r + 1] << 16) | (u64)tile[(2 * r + 2) & 31];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        u32 word = 0;
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int p = 2 * j + e;
            const u32 st = (u32)(win >> (32 - 4 * p)) & 0xffffu;
            const u32 x = (st + 1u) * st;
            const u32 hsh = (x >> 6) & 511u;
            int v0 = t8[2 * hsh], v1 = t8[2 * hsh + 1];
            if ((x >> 15) & 1u) v0 = -v0;
            word |= ((u32)(v0 & 0xff) | ((u32)(v1 & 0xff) << 8)) << (16 * e);
        }
        w[j] = (int)word;
    }
}
'''

PRE = r'''
// One block per token: sign flip, block Walsh, per-token INT8 quantizer.
extern "C" __global__ void NAME(const XT* x, const i8* su, i8* q, int* q4, float* scale, int M, int K, int ldq, int rad) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int T = blockDim.x, t = threadIdx.x, tok = blockIdx.x;
    const size_t base = (size_t)tok * K;
    float mx = 0.f;
    block_walsh(buf, K, rad, [&](int i) { return ldx(x, base + i) * (float)su[i]; }, [&](int i, float v) { buf[i] = v; mx = fmaxf(mx, fabsf(v)); });
    quantize_store(buf, mx, red, q, q4, scale, tok, K, ldq);
}
'''

POST = r'''
// One block per token: z[i] = SCALE of column col0 + i of the accumulator row (leading dimension lda), block Walsh, sign, bias, BF16 out.
extern "C" __global__ void NAME(const AT* acc, const float* s, const float* rs, const i8* sv, const float* bias, u16* y, int M, int N, int lda, int col0, int rad) {
    extern __shared__ float buf[];
    const int T = blockDim.x, t = threadIdx.x, tok = blockIdx.x;
    const size_t base = (size_t)tok * lda + col0;
    u16* yo = y + (size_t)tok * N;
    block_walsh(buf, N, rad, [&](int i) { return SCALE(acc[base + i], s[tok], rs[i]); },
                [&](int i, float v) { v = v * (float)sv[i]; if (bias) v = v + bias[i]; yo[i] = f2bf(v); });
}

// Three layers that share the accumulator row (q / k / v): columns [0, n0), [n0, n0 + n1), [n0 + n1, n0 + n1 + n2); rs / sv / bias concatenated.
extern "C" __global__ void NAME3(const AT* acc, const float* s, const float* rs, const i8* sv, const float* bias, u16* y0, u16* y1, u16* y2, int M, int n0, int n1, int n2, int lda, int rad) {
    extern __shared__ float buf[];
    const int T = blockDim.x, t = threadIdx.x, tok = blockIdx.x;
    int c0 = 0;
    #pragma unroll
    for (int seg = 0; seg < 3; ++seg) {
        const int N = seg == 0 ? n0 : seg == 1 ? n1 : n2;
        u16* yo = (seg == 0 ? y0 : seg == 1 ? y1 : y2) + (size_t)tok * N;
        const size_t base = (size_t)tok * lda + c0;
        const float* r = rs + c0; const i8* sg = sv + c0; const float* bb = bias ? bias + c0 : 0;
        block_walsh(buf, N, rad, [&](int i) { return SCALE(acc[base + i], s[tok], r[i]); },
                    [&](int i, float v) { v = v * (float)sg[i]; if (bb) v = v + bb[i]; yo[i] = f2bf(v); });
        c0 += N;
    }
}

// Up to three layers sharing the accumulator row, one block per (token, layer): for few tokens (action expert) the layers run on
// different SMs at the same time instead of one after the other. gelu0: apply GELU to the first layer's output (gate of a gated MLP).
extern "C" __global__ void NAMEG(const AT* acc, const float* s, const float* rs, const i8* sv, const float* bias, const u16* lut, u16* y0, u16* y1, u16* y2, int M, int n0, int n1, int n2, int lda, int gelu0, int rad) {
    extern __shared__ float buf[];
    const int tok = blockIdx.x, seg = blockIdx.y;
    const int N = seg == 0 ? n0 : seg == 1 ? n1 : n2, c0 = seg == 0 ? 0 : seg == 1 ? n0 : n0 + n1;
    u16* yo = (seg == 0 ? y0 : seg == 1 ? y1 : y2) + (size_t)tok * N;
    const size_t base = (size_t)tok * lda + c0;
    const float* r = rs + c0; const i8* sg = sv + c0; const float* bb = bias ? bias + c0 : 0;
    const bool gl = gelu0 && seg == 0;
    block_walsh(buf, N, rad, [&](int i) { return SCALE(acc[base + i], s[tok], r[i]); },
                [&](int i, float v) { v = v * (float)sg[i]; if (bb) v = v + bb[i]; u16 b = f2bf(v); if (gl) b = gelu_bf16(b, lut); yo[i] = b; });
}

// Middle of a gated MLP (Gemma: down(gelu(gate(x)) * up(x))): the gate | up accumulator row (columns [0, N) and [N, 2N)) -> scale,
// block Walsh, sign, BF16 -> GELU by table (lut: all 65536 BF16 inputs, produced by PyTorch on this device) -> BF16 product -> sign
// flip, block Walsh and INT8 quantizer of the down projection's input. Shared memory: N floats + N BF16.
extern "C" __global__ void NAMEM2(const AT* acc, const float* s, const float* rs, const i8* sv, const u16* lut, const i8* su, i8* q, int* q4, float* scale, int M, int N, int lda, int ldq, int rad) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    u16* g = (u16*)(buf + N);
    const int T = blockDim.x, t = threadIdx.x, tok = blockIdx.x;
    const size_t base = (size_t)tok * lda;
    block_walsh(buf, N, rad, [&](int i) { return SCALE(acc[base + i], s[tok], rs[i]); }, [&](int i, float v) { g[i] = gelu_bf16(f2bf(v * (float)sv[i]), lut); });
    block_walsh(buf, N, rad, [&](int i) { return SCALE(acc[base + N + i], s[tok], rs[N + i]); },
                [&](int i, float v) { const u16 u = f2bf(v * (float)sv[N + i]); buf[i] = bf2f(f2bf(bf2f(g[i]) * bf2f(u))) * (float)su[i]; });
    float mx = 0.f;
    block_walsh(buf, N, rad, [&](int i) { return buf[i]; }, [&](int i, float v) { buf[i] = v; mx = fmaxf(mx, fabsf(v)); });
    quantize_store(buf, mx, red, q, q4, scale, tok, N, ldq);
}

// Middle of a plain MLP (SigLIP: fc2(gelu(fc1(x)))): accumulator row -> scale, block Walsh, sign, bias, BF16 -> GELU table -> sign flip,
// block Walsh and INT8 quantizer of fc2's input.
extern "C" __global__ void NAMEM1(const AT* acc, const float* s, const float* rs, const i8* sv, const float* bias, const u16* lut, const i8* su, i8* q, int* q4, float* scale, int M, int N, int lda, int ldq, int rad) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int T = blockDim.x, t = threadIdx.x, tok = blockIdx.x;
    const size_t base = (size_t)tok * lda;
    block_walsh(buf, N, rad, [&](int i) { return SCALE(acc[base + i], s[tok], rs[i]); },
                [&](int i, float v) { v = v * (float)sv[i]; if (bias) v = v + bias[i]; buf[i] = bf2f(gelu_bf16(f2bf(v), lut)) * (float)su[i]; });
    float mx = 0.f;
    block_walsh(buf, N, rad, [&](int i) { return buf[i]; }, [&](int i, float v) { buf[i] = v; mx = fmaxf(mx, fabsf(v)); });
    quantize_store(buf, mx, red, q, q4, scale, tok, N, ldq);
}
'''

PRE2 = r'''
// Second half of a split gated-MLP middle: x = BF16(g * u) (g = GELU(gate), u = up, both BF16), then sign flip, block Walsh, INT8 quantizer.
extern "C" __global__ void k_pre2(const u16* g, const u16* u, const i8* su, i8* q, int* q4, float* scale, int M, int K, int ldq, int rad) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int tok = blockIdx.x;
    const size_t base = (size_t)tok * K;
    float mx = 0.f;
    block_walsh(buf, K, rad, [&](int i) { return bf2f(f2bf(bf2f(g[base + i]) * bf2f(u[base + i]))) * (float)su[i]; },
                [&](int i, float v) { buf[i] = v; mx = fmaxf(mx, fabsf(v)); });
    quantize_store(buf, mx, red, q, q4, scale, tok, K, ldq);
}
'''

PREH = r'''
// W2 (no activation quantization): rotated activations as BF16. h (M, K) row-major when h != 0; hT (K, ldh) (token = column) when hT != 0.
extern "C" __global__ void NAME(const XT* x, const i8* su, u16* h, u16* hT, int M, int K, int ldh, int rad) {
    extern __shared__ float buf[];
    const int T = blockDim.x, t = threadIdx.x, tok = blockIdx.x;
    const size_t base = (size_t)tok * K;
    block_walsh(buf, K, rad, [&](int i) { return ldx(x, base + i) * (float)su[i]; }, [&](int i, float v) { const u16 b = f2bf(v); if (h) h[base + i] = b; if (hT) hT[(size_t)i * ldh + tok] = b; });
}
'''

DECODE = r'''
// Dense BF16 decode for the float path: tb (512, 2) holds the BF16 bit patterns of the INT8 table values.
extern "C" __global__ void k_decode_bf16(const u16* bits, const u16* tb, u16* out, int m, int n) {
    const int tiles = n >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)m * tiles) return;
    const int row = (int)(idx / tiles), j = (int)(idx % tiles), r = row & 15;
    const u16* tile = bits + ((size_t)(row >> 4) * tiles + j) * 32;
    const u64 win = ((u64)tile[2 * r] << 32) | ((u64)tile[2 * r + 1] << 16) | (u64)tile[(2 * r + 2) & 31];
    u16* o = out + (size_t)row * n + 16 * j;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const u32 st = (u32)(win >> (32 - 4 * p)) & 0xffffu;
        const u32 x = (st + 1u) * st;
        const u32 hsh = (x >> 6) & 511u;
        o[2 * p] = tb[2 * hsh] ^ (u16)(((x >> 15) & 1u) << 15);
        o[2 * p + 1] = tb[2 * hsh + 1];
    }
}

// Dense INT8 decode: out (m, n) row-major, one thread per (row, tile column): 16 weights.
extern "C" __global__ void k_decode(const u16* bits, const i8* t8, int* out, int m, int n) {
    const int tiles = n >> 4;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)m * tiles) return;
    const int row = (int)(idx / tiles), j = (int)(idx % tiles);
    int w[4];
    decode_row(bits + ((size_t)(row >> 4) * tiles + j) * 32, row & 15, t8, w);
    int* o = out + (size_t)row * (n >> 2) + 4 * j;
    o[0] = w[0]; o[1] = w[1]; o[2] = w[2]; o[3] = w[3];
}
'''

GEMM = r'''
// Small-M products for the action expert (MT tokens, compile-time). One warp per output row, 16 rows (one tile row) per block; the INT8
// activations sit in shared memory as (K/4, MT) words; z (MT, N) FP32 = float(acc) * s[token] * rs[row].
__device__ __forceinline__ void finish(int* acc, int lane, int row, int N, const float* s, const float* rs, float* z) {
    #pragma unroll
    for (int m = 0; m < MT; ++m) {
        int v = acc[m];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) z[(size_t)m * N + row] = ((float)v * s[m]) * rs[row];
    }
}

extern "C" __global__ void k_gemm_w8(const int* W, const int* q4, const float* s, const float* rs, float* z, int N, int K) {
    extern __shared__ int sq[];
    const int T = blockDim.x, t = threadIdx.x, words = K >> 2;
    for (int i = t; i < words * MT; i += T) sq[i] = q4[i];
    __syncthreads();
    const int lane = t & 31, row = blockIdx.x * (T >> 5) + (t >> 5);
    if (row >= N) return;
    int acc[MT];
    #pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0;
    const int* wrow = W + (size_t)row * words;
    for (int j = lane; j < words; j += 32) {
        const int w = wrow[j]; const int* qq = sq + j * MT;
        #pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = __dp4a(w, qq[m], acc[m]);
    }
    finish(acc, lane, row, N, s, rs, z);
}

// Same product with the activations read straight from global memory (no per-block copy into shared memory).
extern "C" __global__ void k_gemm_w8d(const int* W, const int* q4, const float* s, const float* rs, float* z, int N, int K) {
    const int T = blockDim.x, t = threadIdx.x, words = K >> 2;
    const int lane = t & 31, row = blockIdx.x * (T >> 5) + (t >> 5);
    if (row >= N) return;
    int acc[MT];
    #pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0;
    const int* wrow = W + (size_t)row * words;
    for (int j = lane; j < words; j += 32) {
        const int w = wrow[j]; const int* qq = q4 + j * MT;
        #pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = __dp4a(w, qq[m], acc[m]);
    }
    finish(acc, lane, row, N, s, rs, z);
}

// Decode on the fly: block = one tile row (16 output rows = 16 warps), lane = tile column (stride 32).
// Members of a group (q/k/v, gate/up) are concatenated along the rows: tile rows < nb0 use table 0, < nb1 table 1, else table 2.
extern "C" __global__ void k_gemm_tcq(const u16* bits, const i8* t8, const int* q4, const float* s, const float* rs, float* z, int N, int K, int nb0, int nb1) {
    extern __shared__ int sq[];
    __shared__ i8 st8[1024];
    const int T = blockDim.x, t = threadIdx.x, words = K >> 2, tiles = K >> 4;
    for (int i = t; i < words * MT; i += T) sq[i] = q4[i];
    const i8* tab = t8 + 1024 * (((int)blockIdx.x >= nb0) + ((int)blockIdx.x >= nb1));
    for (int i = t; i < 1024; i += T) st8[i] = tab[i];
    __syncthreads();
    const int lane = t & 31, r = t >> 5, row = blockIdx.x * 16 + r;
    int acc[MT];
    #pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0;
    const u16* trow = bits + (size_t)blockIdx.x * tiles * 32;
    for (int j = lane; j < tiles; j += 32) {
        int w[4];
        decode_row(trow + (size_t)j * 32, r, st8, w);
        const int* qq = sq + 4 * j * MT;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            #pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] = __dp4a(w[e], qq[e * MT + m], acc[m]);
        }
    }
    finish(acc, lane, row, N, s, rs, z);
}
'''

GEMMF = r'''
// W2 small-M products: BF16 rotated activations hT (K, MT) in shared memory, INT8 weights, FP32 accumulation; z = acc * rs[row].
__device__ __forceinline__ void finish_f(float* acc, int lane, int row, int N, const float* rs, float* z) {
    #pragma unroll
    for (int m = 0; m < MT; ++m) {
        float v = acc[m];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) z[(size_t)m * N + row] = v * rs[row];
    }
}

extern "C" __global__ void k_gemmf_w8(const int* W, const u16* hT, const float* rs, float* z, int N, int K) {
    extern __shared__ u16 sh[];
    const int T = blockDim.x, t = threadIdx.x, words = K >> 2;
    for (int i = t; i < K * MT; i += T) sh[i] = hT[i];
    __syncthreads();
    const int lane = t & 31, row = blockIdx.x * (T >> 5) + (t >> 5);
    if (row >= N) return;
    float acc[MT];
    #pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;
    const int* wrow = W + (size_t)row * words;
    for (int j = lane; j < words; j += 32) {
        const int w = wrow[j];
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float wv = (float)(i8)((w >> (8 * e)) & 0xff); const u16* hh = sh + (4 * j + e) * MT;
            #pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] += wv * bf2f(hh[m]);
        }
    }
    finish_f(acc, lane, row, N, rs, z);
}

extern "C" __global__ void k_gemmf_tcq(const u16* bits, const i8* t8, const u16* hT, const float* rs, float* z, int N, int K, int nb0, int nb1) {
    extern __shared__ u16 sh[];
    __shared__ i8 st8[1024];
    const int T = blockDim.x, t = threadIdx.x, tiles = K >> 4;
    const i8* tab = t8 + 1024 * (((int)blockIdx.x >= nb0) + ((int)blockIdx.x >= nb1));
    for (int i = t; i < K * MT; i += T) sh[i] = hT[i];
    for (int i = t; i < 1024; i += T) st8[i] = tab[i];
    __syncthreads();
    const int lane = t & 31, r = t >> 5, row = blockIdx.x * 16 + r;
    float acc[MT];
    #pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;
    const u16* trow = bits + (size_t)blockIdx.x * tiles * 32;
    for (int j = lane; j < tiles; j += 32) {
        int w[4];
        decode_row(trow + (size_t)j * 32, r, st8, w);
        #pragma unroll
        for (int e = 0; e < 16; ++e) {
            const float wv = (float)(i8)((w[e >> 2] >> (8 * (e & 3))) & 0xff); const u16* hh = sh + (16 * j + e) * MT;
            #pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] += wv * bf2f(hh[m]);
        }
    }
    finish_f(acc, lane, row, N, rs, z);
}
'''

_CACHE = {}
CONFIG = dict(GELU_MODE=0)          # compile-time switch: GELU evaluation in the fused MLP kernels (see pick_gelu)

GELU_TEST = r'''
extern "C" __global__ void k_gelu_test(u16* out, int fma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) out[i] = gelu_formula((u16)i, fma);
}
'''


def pick_gelu():
    """Choose how the fused MLP kernels evaluate GELU: the in-kernel formula if it reproduces PyTorch on all 65536 BF16 inputs
    (NaN inputs excepted: both must be NaN), else the lookup table. Returns the report."""
    if 'gelu_pick' in _CACHE:
        return _CACHE['gelu_pick']
    ks = compile_module(COMMON.replace('GELU_MODE', '1') + GELU_TEST)
    ref = gelu_table()
    x = torch.arange(65536, dtype=torch.int32, device='cuda').to(torch.int16).view(torch.bfloat16)
    finite = ~torch.isnan(x)
    report = {}
    for mode, fma in ((1, 0), (2, 1)):
        out = torch.empty(65536, dtype=torch.bfloat16, device='cuda')
        ks['k_gelu_test'](256, 256, [out, fma])
        same = (out.view(torch.int16) == ref.view(torch.int16)) | (torch.isnan(out) & torch.isnan(ref))
        report[mode] = dict(mismatches=int((~same & finite).sum()), nan_inputs_ok=bool((torch.isnan(out) == torch.isnan(ref))[~finite].all()))
    mode = next((m for m in (1, 2) if report[m]['mismatches'] == 0 and report[m]['nan_inputs_ok']), 0)
    CONFIG['GELU_MODE'] = mode
    _CACHE.pop('base', None)
    _CACHE['gelu_pick'] = dict(mode=mode, report=report)
    return _CACHE['gelu_pick']
SHARED_LIMIT = 48 * 1024


def _variant(template, name, **defs):
    src = template.replace('NAME', name)
    for k, v in defs.items():
        src = src.replace(k, v)
    return src


def _post(at, scale):
    src = '#undef SCALE\n#define SCALE(a, b, c) ' + scale + '\n' + POST
    for k, v in dict(NAMEM2=f'k_mid2_{at[1]}', NAMEM1=f'k_mid1_{at[1]}', NAMEG=f'k_postg_{at[1]}', NAME3=f'k_post3_{at[1]}', NAME=f'k_post_{at[1]}', AT=at[0]).items():
        src = src.replace(k, v)
    return src


def base():
    """pre (BF16 / FP32 input), post / post3 / MLP middles (INT32, FP32 or BF16 accumulator, BF16 output), dense decode."""
    if 'base' not in _CACHE:
        pick_gelu()
        src = COMMON + _variant(PRE, 'k_pre_bf16', XT='u16') + _variant(PRE, 'k_pre_f32', XT='float')
        src += _post(('int', 'i32'), '((((float)(a)) * (b)) * (c))') + _post(('float', 'f32'), '(a)') + _post(('u16', 'bf16'), '(bf2f(a) * (c))')
        src += _variant(PREH, 'k_preh_bf16', XT='u16') + _variant(PREH, 'k_preh_f32', XT='float')
        src += PRE2 + DECODE
        ks = compile_module('#include <sm_61_intrinsics.h>\n' + src, defines=CONFIG)
        _CACHE['wide'] = {}
        for k in ks.values():
            if k.name.startswith('k_mid2'):
                try:
                    k.max_dynamic_shared(16384 * 6 + 256); _CACHE['wide'][k.name] = True
                except RuntimeError:
                    _CACHE['wide'][k.name] = False                 # device cannot hold 96 KB per block: 16384-wide MLPs use the unfused kernels
                    k.max_dynamic_shared(4096 * 6 + 256)
            elif k.name.startswith(('k_pre', 'k_post', 'k_mid1')):          # includes k_pre2, k_postg, k_post3
                k.max_dynamic_shared(16384 * 4 + 256)
        _CACHE['base'] = ks
    return _CACHE['base']


def gelu_table():
    """BF16 GELU (tanh approximation) of every BF16 bit pattern, computed by PyTorch on this device: the fused MLP kernels look the
    activation up instead of re-implementing it, so they reproduce torch.nn.functional.gelu(x, approximate='tanh') exactly."""
    if 'gelu' not in _CACHE:
        x = torch.arange(65536, dtype=torch.int32, device='cuda').to(torch.int16).view(torch.bfloat16)
        _CACHE['gelu'] = torch.nn.functional.gelu(x, approximate='tanh').contiguous()
    return _CACHE['gelu']


def gemm(tokens):
    key = ('gemm', tokens)
    if key not in _CACHE:
        ks = compile_module('#include <sm_61_intrinsics.h>\n' + COMMON + GEMM + GEMMF, defines=dict(MT=tokens))
        for k in ks.values():
            if k.name != 'k_gemm_w8d':
                k.max_dynamic_shared(4096 * tokens * (2 if 'gemmf' in k.name else 1) + 64)
        _CACHE[key] = ks
    return _CACHE[key]


PLAN = {('large', 4304): (128, 4)}                # (tokens class, width) -> (threads per block, radix log2) overrides from the tuning runs


def plan(M, width):
    """Threads per token block and radix (log2 of values per group) of the Walsh passes, from the in-graph tuning on an RTX 6000 Ada
    (dev/t_gbench.py): radix 16 once the top block is >= 1024 wide; threads: one radix-16 group per thread when many tokens keep
    every SM busy (prefill), 128 for a few hundred tokens (vision, per image), 256 for the action expert's ~10 tokens."""
    cls = 'small' if M <= 16 else 'mid' if M <= 512 else 'large'
    if (cls, width) in PLAN:
        return PLAN[(cls, width)]
    top = 1 << (width.bit_length() - 1)
    rad = 4 if top >= 1024 else 2
    if cls == 'large':
        return max(64, min(1024, top // 16)), rad
    if cls == 'mid':
        return 128, rad
    return (256 if top >= 1024 else 128), rad


def pre(x, su, q=None, q4=None, scale=None):
    """x (M, K) BF16 or FP32 contiguous -> q (M, K) int8 and / or q4 (K/4, ld) int32, scale (M,) FP32."""
    M, K = x.shape; T, rad = plan(M, K)
    k = base()['k_pre_bf16' if x.dtype == torch.bfloat16 else 'k_pre_f32']
    k(M, T, [x, su, q, q4, scale, M, K, 0 if q4 is None else q4.shape[1], rad], shared=4 * K)


_AT = {torch.int32: 'i32', torch.float32: 'f32', torch.bfloat16: 'bf16'}


def post(acc, s, rs, sv, bias, y, col0=0):
    """acc (M, lda) int32 (scaled here by s, rs), FP32 (already scaled) or BF16 (scaled by rs); columns col0 .. col0 + N -> y (M, N) BF16."""
    M, N = y.shape; T, rad = plan(M, N)
    base()[f'k_post_{_AT[acc.dtype]}'](M, T, [acc, s, rs, sv, bias, y, M, N, acc.shape[1], col0, rad], shared=4 * N)


def post3(acc, s, rs, sv, bias, ys):
    """q / k / v in one launch: rs, sv, bias concatenated over the three layers; ys = three (M, N_i) BF16 outputs."""
    M = ys[0].shape[0]; n = [y.shape[1] for y in ys]; T, rad = plan(M, max(n))
    base()[f'k_post3_{_AT[acc.dtype]}'](M, T, [acc, s, rs, sv, bias, ys[0], ys[1], ys[2], M, n[0], n[1], n[2], acc.shape[1], rad], shared=4 * max(n))


def postg(acc, s, rs, sv, bias, ys, gelu0=False):
    """Two or three layers sharing the accumulator row, one block per (token, layer) (few tokens); optional GELU on the first."""
    M = ys[0].shape[0]; n = [y.shape[1] for y in ys] + [0] * (3 - len(ys)); T, rad = plan(M, max(n))
    yy = list(ys) + [None] * (3 - len(ys))
    base()[f'k_postg_{_AT[acc.dtype]}']((M, len(ys)), T, [acc, s, rs, sv, bias, gelu_table(), yy[0], yy[1], yy[2], M, n[0], n[1], n[2], acc.shape[1], gelu0, rad], shared=4 * max(n))


def pre2(g, u, su, q=None, q4=None, scale=None):
    M, K = g.shape; T, rad = plan(M, K)
    base()['k_pre2'](M, T, [g, u, su, q, q4, scale, M, K, 0 if q4 is None else q4.shape[1], rad], shared=4 * K)


def mid2_available(N):
    base()
    return N * 6 + 256 <= SHARED_LIMIT or all(_CACHE['wide'].values())


def mid2(acc, s, rs, sv, su, q=None, q4=None, scale=None):
    """Gated-MLP middle: acc (M, 2N) gate | up; rs, sv concatenated (2N); su = input signs of the down projection (N)."""
    M = acc.shape[0]; N = acc.shape[1] // 2; T, rad = plan(M, N)
    base()[f'k_mid2_{_AT[acc.dtype]}'](M, T, [acc, s, rs, sv, gelu_table(), su, q, q4, scale, M, N, acc.shape[1], 0 if q4 is None else q4.shape[1], rad], shared=6 * N)


def mid1(acc, s, rs, sv, bias, su, q=None, q4=None, scale=None):
    """Plain-MLP middle: acc (M, N) of the first layer; su = input signs of the second layer."""
    M, N = acc.shape; T, rad = plan(M, N)
    base()[f'k_mid1_{_AT[acc.dtype]}'](M, T, [acc, s, rs, sv, bias, gelu_table(), su, q, q4, scale, M, N, N, 0 if q4 is None else q4.shape[1], rad], shared=4 * N)


def decode(bits, t8, out):
    m, n = out.shape
    total = m * (n >> 4)
    base()['k_decode']((total + 255) // 256, 256, [bits, t8, out, m, n])


GEMM_PLAN = dict(rows=16, direct=False)        # rows (= warps) per block and activation source of k_gemm_w8 (tuning)


def gemm_w8(W, q4, s, rs, z):
    """W (N, K) int8 resident, q4 (K/4, M) int32, z (M, N) FP32."""
    N, K = W.shape; M = q4.shape[1]; R = GEMM_PLAN['rows']
    if GEMM_PLAN['direct']:
        gemm(M)['k_gemm_w8d']((N + R - 1) // R, 32 * R, [W, q4, s, rs, z, N, K])
    else:
        gemm(M)['k_gemm_w8']((N + R - 1) // R, 32 * R, [W, q4, s, rs, z, N, K], shared=K * M)


def gemm_tcq(bits, t8, q4, s, rs, z, K, bounds=(1 << 30, 1 << 30)):
    M, N = z.shape
    gemm(M)['k_gemm_tcq'](N // 16, 512, [bits, t8, q4, s, rs, z, N, K, bounds[0], bounds[1]], shared=K * M)


# ---- W2 (float activations) helpers
def pre_h(x, su, h=None, hT=None):
    M, K = x.shape; T, rad = plan(M, K)
    k = base()['k_preh_bf16' if x.dtype == torch.bfloat16 else 'k_preh_f32']
    k(M, T, [x, su, h, hT, M, K, 0 if hT is None else hT.shape[1], rad], shared=4 * K)


def decode_bf16(bits, tb, out):
    m, n = out.shape
    total = m * (n >> 4)
    base()['k_decode_bf16']((total + 255) // 256, 256, [bits, tb, out, m, n])


def gemmf_w8(W, hT, rs, z):
    N, K = W.shape; M = hT.shape[1]
    gemm(M)['k_gemmf_w8']((N + 15) // 16, 512, [W, hT, rs, z, N, K], shared=2 * K * M)


def gemmf_tcq(bits, t8, hT, rs, z, K, bounds=(1 << 30, 1 << 30)):
    M, N = z.shape
    gemm(M)['k_gemmf_tcq'](N // 16, 512, [bits, t8, hT, rs, z, N, K, bounds[0], bounds[1]], shared=2 * K * M)
