import io

p = 'src/cuda/PdlpKernels.cu'
s = io.open(p, encoding='utf-8').read()
add = io.open('.adaptive_cu.txt', encoding='utf-8').read()
anchor = '} // namespace gpu'
idx = s.rindex(anchor)
s = s[:idx] + add + s[idx:]
io.open(p, 'w', encoding='utf-8').write(s)
print('ok: adaptive kernels appended to PdlpKernels.cu')
