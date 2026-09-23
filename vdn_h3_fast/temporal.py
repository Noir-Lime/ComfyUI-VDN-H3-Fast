"""CuTe five-tap temporal shift with eager BF16 rounding and no retained tensors."""
import functools

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_tensor
import cuda.bindings.driver as cuda


@cute.kernel
def _shift_kernel(x: cute.Tensor, w: cute.Tensor, y: cute.Tensor,
                  n: cutlass.Constexpr, frame: cutlass.Constexpr,
                  channels: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    row = cute.assume((block * 128 + tid) * 8, divby=8)
    dtype = y.element_type
    if row < n:
        channel = cute.assume(row % channels, divby=8)
        acc = cute.make_rmem_tensor((8,), cutlass.Float32)
        for tap in cutlass.range_constexpr(5):
            values = cute.make_rmem_tensor((8,), dtype)
            values.fill(0)
            src = cute.assume(row + (tap - 2) * frame, divby=8)
            if src >= 0 and src < n:
                tile = cute.make_tensor(x.iterator + src, cute.make_layout((8,)))
                cute.autovec_copy(tile, values)
            weights = cute.make_rmem_tensor((8,), dtype)
            tile_w = cute.make_tensor(w.iterator + tap * channels + channel, cute.make_layout((8,)))
            cute.autovec_copy(tile_w, weights)
            for lane in cutlass.range_constexpr(8):
                product = values[lane].to(cutlass.Float32) * weights[lane].to(cutlass.Float32)
                part = cutlass.BFloat16(cute.arch.cvt_f32_bf16(product.ir_value())).to(cutlass.Float32)
                if cutlass.const_expr(tap == 0):
                    acc[lane] = part
                else:
                    total = acc[lane] + part
                    acc[lane] = cutlass.BFloat16(cute.arch.cvt_f32_bf16(total.ir_value())).to(cutlass.Float32)
        result = cute.make_rmem_tensor((8,), dtype)
        result.store(acc.load().to(dtype))
        tile_y = cute.make_tensor(y.iterator + row, cute.make_layout((8,)))
        cute.autovec_copy(result, tile_y)


@cute.jit
def _launch(x: cute.Tensor, w: cute.Tensor, y: cute.Tensor,
            n: cutlass.Constexpr, frame: cutlass.Constexpr,
            channels: cutlass.Constexpr, stream: cuda.CUstream):
    _shift_kernel(x, w, y, n, frame, channels).launch(
        grid=((n + 1023) // 1024, 1, 1), block=(128, 1, 1), stream=stream)


@functools.lru_cache(maxsize=8)
def _compiled_shift(n, frame, channels, device):
    with torch.cuda.device(device):
        x = make_fake_tensor(cutlass.BFloat16, (n,), (1,), assumed_align=16)
        w = make_fake_tensor(cutlass.BFloat16, (5 * channels,), (1,), assumed_align=16)
        y = make_fake_tensor(cutlass.BFloat16, (n,), (1,), assumed_align=16)
        return cute.compile(_launch, x, w, y, n, frame, channels, cuda.CUstream(0))


def temporal_shift(x, w):
    with torch.cuda.device(x.device):
        compiled = _compiled_shift(x.numel(), x.shape[1] * x.shape[2], x.shape[2], x.device.index)
        wt = w.T.contiguous()
        out = torch.empty_like(x)
        compiled(from_dlpack(x.view(-1), assumed_align=16),
                 from_dlpack(wt.view(-1), assumed_align=16),
                 from_dlpack(out.view(-1), assumed_align=16),
                 cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out
