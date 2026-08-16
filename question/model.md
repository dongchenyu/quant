VLLM中的模型调用路径:
1. 模型入口,以DeepseekV4举例,路径为vllm/models/deepseek_v4/model.py

入口为class DeepseekV4ForCausalLM,其中可以看到参数config使用的是vllm_config.model_config.hf_config, 这里用的应该是hugging face上的config?
接下来可以看到forward非常少,应该只起了一个定义的作用,不是在这里进行具体计算.只是一个self.model -> self.model_cls -> DeepseekV4Model.
这里是第一跳,接下来跳到了DeepSeekV4Model这个类里面.

```
class DeepseekV4ForCausalLM(
    nn.Module, SupportsPP, SupportsEagle3, DeepseekV4MixtureOfExperts
):
model_cls = DeepseekV4Model
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    super().__init__()
    config = vllm_config.model_config.hf_config
    self.config = config
    self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
    return hidden_states
```

进入到DeekSeekV4Model类里面,可以看到config来源,量化参数quant_config是vllm_config提供的,config依旧由vllm_config.model_config.hf_config提供.

```
class DeepseekV4Model(nn.Module, EagleModelMixin):
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    self.config = config
    self.quant_config = quant_config
    self.parallel_config = vllm_config.parallel_config
    self.start_layer, self.end_layer, self.layers = make_layers(
        config.num_hidden_layers,
        lambda prefix: DeepseekV4DecoderLayer(
            vllm_config,
            prefix=prefix,
            topk_indices_buffer=self.topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        ),
        prefix=f"{prefix}.layers",
    )
```


这个forward里面包括了embedding的内容,这里面可以清楚的看到embedding之后直接写死了hidden_state就是self.hc_mult条stream
```
def forward(self,input_ids: torch.Tensor,positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
```

接下来进入DeepseekV4DecoderLayer,看下DecoderLayer具体是怎么实现的. 这里面第一层只执行pre,然后接attention,后面是post-pre+FFN/attention,相对应的,最后一层只有post.
最后通过hc_head_fused_kernel_tilelang将多个通道进行汇总.
```
class DeepseekV4DecoderLayer(nn.Module):
    def forward(self, x: torch.Tensor, positions: torch.Tensor, input_ids: torch.Tensor | None, post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None, residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            # Run standalone mhc_pre on first layer
            residual = x
            post_mix, res_mix, x = mhc_pre_tilelang(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.rms_norm_eps,
                self.hc_eps, self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters, norm_weight=attn_norm_weight,norm_eps=attn_norm_eps,
            )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(x, residual, post_mix, res_mix,
                self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.rms_norm_eps, self.hc_eps, self.hc_eps,
                self.hc_post_alpha, self.hc_sinkhorn_iters, n_splits=1, tile_n=1, norm_weight=attn_norm_weight, norm_eps=attn_norm_eps,
            )
        # attn_norm is fused into mhc_pre_tilelang / mhc_fused_post_pre above.
        x = self.attn(positions, x, None)

    hidden_states = hc_head_fused_kernel_tilelang(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.rms_norm_eps,self.hc_eps,)
```


汇总一下大概是这样
DeepseekV4ForCausalLM.forward ->
self.model(...) ->
DeepseekV4Model.forward ->
Embedding ->
repeat [T,hc_mult,H] ->
for DeepseekV4DecoderLayer ->
        
第一层：->
    mhc_pre_tilelang ->
    Attention ->
    mhc_fused_post_pre_tilelang ->
    MoE/FFN ->

第二层以后：
    mhc_fused_post_pre_tilelang ->
    Attention ->
    mhc_fused_post_pre_tilelang ->
    MoE/FFN ->
...
mhc_post_tilelang ->
hc_head_fused_kernel_tilelang ->
RMSNorm ->
output ->


然后上面model中涉及到的具体算子的调用就在tilelang.py以及tilelang_kernel.py里找到了.