# RMSNorm Backward

这份文档解释 RMSNorm backward 要算什么，以及当前项目已经准备好的 reference。

当前还没有 CUDA backward kernel。现在完成的是：

```text
纯 PyTorch forward reference
手写 backward formula reference
backward reference tests
gradcheck
```

这些东西是后面写 CUDA backward 的对照答案。

## 1. Forward 回顾

RMSNorm forward 是：

```text
mean_square = mean(x * x)
inv_rms = 1 / sqrt(mean_square + eps)
y = x * inv_rms * weight
```

这里所有计算都沿最后一维 `hidden_size` 做。

如果输入是：

```text
x: [batch, seq_len, hidden_size]
```

那么每个 token 的 hidden 向量单独算一个 `inv_rms`。

## 2. Backward 要求什么

训练时，上游会给我们：

```text
grad_out = dL/dy
```

我们要算：

```text
dx      = dL/dx
dweight = dL/dweight
```

对于 fused add + RMSNorm：

```text
y = RMSNorm(x + residual, weight)
```

还要算：

```text
dresidual = dL/dresidual
```

因为 `tmp = x + residual`，所以：

```text
dx = dtmp
dresidual = dtmp
```

## 3. dweight 怎么算

forward 里：

```text
y[i] = x[i] * inv_rms * weight[i]
```

所以对 `weight[i]` 的梯度是：

```text
dweight[i] = sum_over_rows(grad_out[i] * x[i] * inv_rms)
```

通俗理解：

```text
weight 的每个位置服务所有 row
所以 dweight 要把所有 row 上同一个 hidden 位置的贡献加起来
```

## 4. dx 怎么算

如果只看 `y = x * inv_rms * weight`，似乎：

```text
dx = grad_out * weight * inv_rms
```

但 `inv_rms` 本身也是由整行 `x` 算出来的，所以还要补一项。

最终公式：

```text
scaled_grad = grad_out * weight
dot = sum(scaled_grad * x)
dx = inv_rms * scaled_grad - x * inv_rms^3 * dot / hidden_size
```

通俗理解：

```text
第一项：y 直接依赖 x
第二项：x 改变了整行的 rms，整行缩放系数也跟着变
```

## 5. 项目里的 reference

源码：

```text
rmsnorm_cuda/reference.py
```

里面有：

```python
rmsnorm_reference(...)
fused_add_rmsnorm_reference(...)
rmsnorm_backward_reference(...)
fused_add_rmsnorm_backward_reference(...)
```

测试：

```text
tests/test_backward_reference.py
```

它做三件事：

```text
1. 手写 RMSNorm backward 对比 PyTorch autograd
2. 手写 fused backward 对比 PyTorch autograd
3. 用 gradcheck 检查 reference forward 的梯度
```

## 6. 后续 CUDA backward 怎么写

CUDA backward 大概率会分两类输出：

```text
dx / dresidual
dweight
```

难点是 `dweight`：

```text
它要跨 rows 汇总
多个 block 会同时贡献同一个 hidden 位置
需要 reduction
```

第一版建议不要追求极致性能，可以先写成两步：

```text
1. 每个 row 算 dx，同时写出 partial dweight
2. 第二个 kernel 把 partial dweight 沿 rows reduce 成最终 dweight
```

这样代码更容易验证。等正确性稳定后，再考虑把 `dweight` reduction 优化掉。

## 7. 下一步

建议下一步实现：

```text
rmsnorm_backward(x, weight, grad_out, eps) -> dx, dweight
```

先只支持：

```text
float32
contiguous tensor
forward 输入 [..., hidden_size]
```

通过测试后，再扩展到：

```text
float16 / bfloat16
fused add + RMSNorm backward
更快的 dweight reduction
```
