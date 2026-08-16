- Q52: 介绍一下 calibration？为什么要做 calibration？（要点：只针对 activation 而非 weight，activation 动态范围大需真实数据前向一遍统计范围）
calibration 是静态量化的时候使用的一种方法.具体方法是选择一批有代表性的数据当作输入数据进行前向推理,根据这些输入得到的每层的结果来确定每一层量化的scale和zero-point.
之后有其他数据输入的时候,量化参数一概使用之前得到的scale和zero-point.
因为activation是动态的,输入不同,得到的activation就不同,所以无法提前知道它的范围,跟weight不一样,所以就先选用一些有代表性的数据,相当于把activation的数据范围也提前拿到,
这样就可以提前固定下来scale和zero-point,避免之后再计算,节省计算成本.

- Q53: 关于 calibration 你知道哪些算法？（MinMax 与 KL 散度法的流程差异，KL 如何通过直方图选截断阈值来抑制 outlier）
MinMax、Percentile(百分位截断)、KL Divergence(KL 散度)
他们的共同目标都是在activation值中选择最大值max和最小值min,用来计算zero-point和scale. 当然计算方法各有不同.
MinMax: 选当前激活值中的最大值max和最小值min,直接用max和min计算zero-point和scale.
Percentile(百分位截断): 取一个超参数a,然后保留中间a%的数据,两侧的极值截断,如果是[0, 255]这种非对称量化通常是截断右侧偏大的极值,这种方法可以更好的对抗outlier的影响.
KL Divergence(KL 散度): KL散度法会先选择两端(对称量化)或者最大值一端(非对称量化)的截断阈值,具体方法为将量化后的值与每个值对应的样本数量绘制成直方图,其中横轴表示量化后的值,
纵轴表示每个值对应的样本数量,然后遍历每一个量化值作为截断值.截断值右侧(非对称量化如uint8)或者两侧(对称量化如int8)的值都量化成被选定的截断值,然后通过公式来计算散度,即数据的失真程度.选择一个散度最小的截断值来当最终的截断值,大于(或者绝对值大于)该截断值的都量化成那个截断值.
KL散度公式如下:
$D_{KL}(P \parallel Q) = \sum_i p_i \log \frac{p_i}{q_i}$

- Q76: 大模型量化主要量化哪些算子？为什么不量化其它算子？（从时间占比和精度敏感度两个维度回答，说明 softmax/layernorm 为何保持高精度）
大模型量化主要量化哪些算子？
GEMM/GEMV,因为计算量和参数量都高,收益明显. 计算量大通常耗时更长,量化收益更大,而且如果量化成INT8/INT4,通常硬件还有专门的加速路径.
同时GEMM/GEMV是线性计算,精度敏感度不像指数型的softmax那样敏感.

为什么不量化其它算子？
softmax算子中有指数计算,对误差比较敏感,因为误差经过指数传播之后会变的很大.
Layernorm有标准差的计算,涉及平方和reduce,单个元素的误差可能影响整行.
而且softmax和Layernorm的参数量极低,且计算量不大,量化收益很低.

- Q54: 量化所产生的精度问题，你一般是什么解决思路或者采用什么解决办法？
首先使用fp32模型做baseline,对每一层单独做量化,查看哪一层的量化误差明显偏高,对于量化误差偏高的层,按照下面的方法尝试.
(1) 如果是使用per-tensor,可以考虑换成per-channel或者per-group降低精度误差,这个通常是解决outlier的问题,比如outlier出现在某个channel,
这么干可以让没有outlier的channel保持正常.
(2) 如果是因为使用minmax方法,又碰到outlier值影响了量化精度,那么通常要考虑使用Percentile、KL Divergence 等方法降低outlier值的影响
(3) 将部分对量化误差比较敏感,计算量/参数又不是很大的酸子,比如LayerNorm/RMSNorm/Softmax, 可以考虑调整回高精度的计算,比如FP16/FP32/BF16等等.
(4) 如果是采用的量化方法是PTQ,在条件允许重新训练的情况下可以考虑改成QAT

- Q50: 什么是 GPU 的 occupancy？哪些因素会影响到它的大小？
一个SM上的active warp数量与SM硬件理论支持的warp数量的比值.
Theoretical Occupancy: 通过静态资源计算出来的Occupancy,是理论值
Achieved Occupancy: 运行过程中的实测结果
如果grid太小,没有填满硬件,可能导致Achieved Occupancy低于Theoretical Occupancy
哪些因素会影响到它的大小？
(1)register,每个SM上register数量固定,大概是65536,如果某个线程用的多了,那么register资源能容纳的warp就少了
(2)shared memory,与register同理,每个SM上shared memory是有限的,如果用的多了,那么shared memory能容纳的warp就少了
(3)block的线程分配,如果每个SM上最多能容纳64个warp(2048 thread),那最好每个block的线程数可以被2048整除,如果是768这种,那么只能放2个block,会造成浪费

- Q64: 随着 seqlen 增加，encoder / decoder 的计算量、访存量、计算密度分别怎么变化？
这里面涉及的运算是GEMV和GEMM,假设GEMM的A矩阵为 M * K,B矩阵为 K * N, 那这样的话seqlen = M
计算量: 2MKN (mul+add) 
访存量: MK + KN + MN
计算密度: 2MKN / (MK + KN + MN)

decoder:
其中seqlen = M = N
Prefill: 
计算量: 2 * M * M * K
访存量: M * M + 2 * M * K
计算密度: 2MK / (M + 2K)
如果M远小于K,那么随着M增长,计算密度线性增加.
如果M远大于K,那么计算密度趋近于2K.

Decode: decode每次只是输入1个token,更接近GEMV,此时M表示KV cache里的历史序列长度
计算量: (2MK) + (2MK) (QK + (QK)V)
访存量: (MK + K + M) +  (MK + K + M)(QK + (QK)V)
计算密度: 2MK / (MK + K + M)
因为这个时候M通常不是很小的值,所以计算密度趋近于2.

encoder类似于decoder的prefill,只不过最后乘V的时候,V的N维度是一个长度很大且固定的值,值的大小为整个词表的size


- Q45: register spill 是什么意思？发生了 register spill 是好是坏？为什么？
某个cuda核函数中每个线程使用的寄存器太多了,导致一部分本来应该存放在寄存器的变量转移到了local memory,通常都是坏事,因为local memory访问很慢,
通常要几百个周期. 不过一般情况下是会先减少每个SM的block.有几种情况例外:每个SM上就一个block,但是寄存器依旧不够用. 编译参数有限制寄存器的最大使用. 
某个线程使用的实在太多(我记得这个有上限,是255?)
不过极端情况下减少寄存器的使用,用少数local memory换取更高的occupancy也可能更好(这个我觉得存在,但是很少,local memory和寄存器之间性能差距太大了)