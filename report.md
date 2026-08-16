不加离群值效果如下,可以看到的是非对称明显好于对称,因为是全正的数值,对称量化中至少一半的quant被浪费掉了所以非对称更好,这里per-tensor和per-channel
的差别不是很明显,但是不是没有,从非对称可以看出来还是有区别的
```
X = torch.rand(4096, 4096) * 2 + 3
============ symmetric + per-tensor ============
mse:
tensor(0.0001)
max_error:
tensor(0.0197)
scale:
tensor(0.0394)
============ asymmetric + per-tensor ============
mse:
tensor(5.1270e-06)
max_error:
tensor(0.0039)
scale:
tensor(0.0078)
============ symmetric + per-channel ============
mse:
tensor(0.0001)
max_error:
tensor(0.0197)
scale:
tensor([[0.0394, 0.0394, 0.0394,  ..., 0.0394, 0.0394, 0.0394]])
============ asymmetric + per-channel ============
mse:
tensor(5.1214e-06)
max_error:
tensor(0.0039)
scale:
tensor([[0.0078, 0.0078, 0.0078,  ..., 0.0078, 0.0078, 0.0078]])
PASS
```

接下来设置了一些离群值
```
X[:, 0] *= 10
X[:, 1] *= 20
X[:, 2] *= 50
X[:, 3] *= 100
```

然后结果如下:
```
============ symmetric + per-tensor ============
mse:
tensor(0.3383)
max_error:
tensor(1.9684)
scale:
tensor(3.9370)
============ asymmetric + per-tensor ============
mse:
tensor(0.3280)
max_error:
tensor(0.9745)
scale:
tensor(1.9490)
============ symmetric + per-channel ============
mse:
tensor(0.0005)
max_error:
tensor(1.9683)
scale:
tensor([[0.3936, 0.7874, 1.9683,  ..., 0.0394, 0.0394, 0.0394]])
============ asymmetric + per-channel ============
mse:
tensor(2.1269e-05)
max_error:
tensor(0.3921)
scale:
tensor([[0.0784, 0.1567, 0.3920,  ..., 0.0078, 0.0078, 0.0078]])
PASS
```

这个差距非常明显,应该说per-channel明显好于per-tensor,因为per=channel是每个channel取一个scale,所以当某些channel的值很极端的时候只影响它自己那个channel,不会影响
整个tensor,否则整个tensor的scale都取决于那个离群值,而大量正常值被压缩在某几个点上,造成量化误差的增加.

后续的LLM.int8()应该是发现了异常值通常都来自极少数channel.
