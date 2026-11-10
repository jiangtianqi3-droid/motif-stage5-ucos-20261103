# 路线 A 全量实验报告

生成时间：2026-11-09T18:21:38.159708+08:00。运行编号：`f20ab88ff89d82d8506c`。

本轮检验统一边权、结构扩召回与分差触发式大模型精排的实际效果。提升与下降均按同一批查询保留。

## 数学修复与实验边界

统一边权为 $w_q(u,r,v)=\phi(\rho_q(r))(1+d_{out}(u))^{-0.25}(1+d_{r,out}(u))^{-0.5}(1+d_{out}(v))^{-0.25}$，再除以全图边权均值。沿用 E3a 的支持加权 RW 与排他度映射。

发送端全局度因子在前向行归一化中依然会抵消，本轮不声称改变这一恒等式；新增的关系局部度与接收端度因子在邻边之间变化时可以保留在转移概率中。所有邻边因子恰好相同时，它们也会抵消。反向矩阵重新按目的端行归一化。

结构证据分别对路径权重、转移概率乘积和 PPR 交汇分数做查询内对数 Min-Max 归一化，再加权并把最大综合支持锚定为 1；无证据为 0。原始 logit 在查询内线性归一化后添加残差。门控和榜首保护是经验约束，不能保证测试指标零退化。

采用用户确认的答案隔离协议。验证候选只过滤观察事实，测试候选只过滤观察及验证事实。检索、结构评分、LLM 提示与触发均不读取测试答案。只在离线计分时过滤其他测试正例并恢复当前目标。报告中明确统计其与标准先按测试真值过滤再取 Top-30 的差异。

最终排名通过重排候选所占原始 logit 槽位计算；完整保留悲观并列秩。候选之外的分数不变，原始候选分数多重集不变，并逐查询断言池外目标排名不变。这里的 Top-30 是答案隔离候选池，不能冒称精确 Filtered Top-30。

每图最多固定无标签抽样 256 条验证事实，双向用于网格选择。LLM 在其中最多 64 个触发查询校准残差，选择完成后固定再评测测试集。Grail 历史协议合并了 inductive valid 与 test，本轮主表只汇报官方 test，因此不能把本表基准与历史合并主表直接当作同一测试集比较。

B0 与 T-4D 分别运行 M0 和 R1–R5。R1 无权、R2 历史度权、R3 历史 E3a、R4 统一边权。FBIngram_25 跳过 LLM。置信区间按原始事实共同重采两方向，2,000 次；不按方向假设独立。

## 标准全实体 Filtered 指标

|图谱|冻结基线|方法|查询数|MRR|Hits@1|Hits@3|Hits@10|
|---|---|---|---:|---:|---:|---:|---:|
|FB15k237Inductive_v1|B0|M0|410|0.490302|0.392683|0.553659|0.678049|
|FB15k237Inductive_v1|B0|R1|410|0.490302|0.392683|0.553659|0.678049|
|FB15k237Inductive_v1|B0|R2|410|0.486409|0.385366|0.553659|0.665854|
|FB15k237Inductive_v1|B0|R3|410|0.491635|0.395122|0.556098|0.675610|
|FB15k237Inductive_v1|B0|R4|410|0.486863|0.382927|0.558537|0.678049|
|FB15k237Inductive_v1|B0|R5|410|0.486863|0.382927|0.558537|0.678049|
|FB15k237Inductive_v1|T-4D|M0|410|0.486464|0.382927|0.556098|0.678049|
|FB15k237Inductive_v1|T-4D|R1|410|0.486082|0.382927|0.553659|0.678049|
|FB15k237Inductive_v1|T-4D|R2|410|0.490167|0.390244|0.553659|0.680488|
|FB15k237Inductive_v1|T-4D|R3|410|0.489823|0.387805|0.556098|0.682927|
|FB15k237Inductive_v1|T-4D|R4|410|0.483116|0.390244|0.536585|0.668293|
|FB15k237Inductive_v1|T-4D|R5|410|0.483116|0.390244|0.536585|0.668293|
|WN18RRInductive_v1|B0|M0|376|0.665709|0.606383|0.699468|0.771277|
|WN18RRInductive_v1|B0|R1|376|0.678406|0.625000|0.696809|0.789894|
|WN18RRInductive_v1|B0|R2|376|0.673352|0.619681|0.694149|0.781915|
|WN18RRInductive_v1|B0|R3|376|0.679577|0.627660|0.691489|0.787234|
|WN18RRInductive_v1|B0|R4|376|0.673871|0.619681|0.696809|0.779255|
|WN18RRInductive_v1|B0|R5|376|0.673871|0.619681|0.696809|0.779255|
|WN18RRInductive_v1|T-4D|M0|376|0.668386|0.606383|0.707447|0.784574|
|WN18RRInductive_v1|T-4D|R1|376|0.684564|0.627660|0.715426|0.800532|
|WN18RRInductive_v1|T-4D|R2|376|0.673370|0.611702|0.710106|0.789894|
|WN18RRInductive_v1|T-4D|R3|376|0.684896|0.627660|0.704787|0.797872|
|WN18RRInductive_v1|T-4D|R4|376|0.679353|0.617021|0.712766|0.797872|
|WN18RRInductive_v1|T-4D|R5|376|0.682411|0.622340|0.715426|0.797872|
|NELLInductive_v1|B0|M0|200|0.667684|0.590000|0.685000|0.870000|
|NELLInductive_v1|B0|R1|200|0.720632|0.660000|0.710000|0.915000|
|NELLInductive_v1|B0|R2|200|0.629740|0.550000|0.640000|0.825000|
|NELLInductive_v1|B0|R3|200|0.715434|0.645000|0.735000|0.870000|
|NELLInductive_v1|B0|R4|200|0.729051|0.680000|0.725000|0.855000|
|NELLInductive_v1|B0|R5|200|0.729120|0.680000|0.725000|0.855000|
|NELLInductive_v1|T-4D|M0|200|0.653899|0.590000|0.665000|0.805000|
|NELLInductive_v1|T-4D|R1|200|0.648232|0.575000|0.645000|0.855000|
|NELLInductive_v1|T-4D|R2|200|0.671887|0.615000|0.660000|0.830000|
|NELLInductive_v1|T-4D|R3|200|0.711411|0.645000|0.740000|0.805000|
|NELLInductive_v1|T-4D|R4|200|0.643285|0.545000|0.680000|0.815000|
|NELLInductive_v1|T-4D|R5|200|0.643285|0.545000|0.680000|0.815000|
|FBIngram_25|B0|M0|11432|0.383840|0.263821|0.433170|0.639871|
|FBIngram_25|B0|R1|11432|0.386836|0.267670|0.434832|0.637334|
|FBIngram_25|B0|R2|11432|0.386676|0.267057|0.434745|0.636109|
|FBIngram_25|B0|R3|11432|0.386127|0.266795|0.434395|0.637334|
|FBIngram_25|B0|R4|11432|0.385768|0.266445|0.434045|0.634797|
|FBIngram_25|T-4D|M0|11432|0.385882|0.266008|0.434395|0.639258|
|FBIngram_25|T-4D|R1|11432|0.387823|0.267845|0.437106|0.637946|
|FBIngram_25|T-4D|R2|11432|0.385933|0.265133|0.435619|0.637246|
|FBIngram_25|T-4D|R3|11432|0.384708|0.264346|0.432208|0.637421|
|FBIngram_25|T-4D|R4|11432|0.388184|0.268719|0.434482|0.636547|

## 相对各自冻结基线的配对差值及 95% 区间

|图谱|基线|方法|指标|差值|下界|上界|
|---|---|---|---|---:|---:|---:|
|FB15k237Inductive_v1|B0|R1|MRR|+0.000000|+0.000000|+0.000000|
|FB15k237Inductive_v1|B0|R1|Hits1|+0.000000|+0.000000|+0.000000|
|FB15k237Inductive_v1|B0|R1|Hits3|+0.000000|+0.000000|+0.000000|
|FB15k237Inductive_v1|B0|R1|Hits10|+0.000000|+0.000000|+0.000000|
|FB15k237Inductive_v1|B0|R2|MRR|-0.003893|-0.016150|+0.009844|
|FB15k237Inductive_v1|B0|R2|Hits1|-0.007317|-0.024390|+0.012195|
|FB15k237Inductive_v1|B0|R2|Hits3|+0.000000|-0.024390|+0.024390|
|FB15k237Inductive_v1|B0|R2|Hits10|-0.012195|-0.026829|+0.002439|
|FB15k237Inductive_v1|B0|R3|MRR|+0.001334|-0.012770|+0.018027|
|FB15k237Inductive_v1|B0|R3|Hits1|+0.002439|-0.014634|+0.024390|
|FB15k237Inductive_v1|B0|R3|Hits3|+0.002439|-0.024390|+0.029268|
|FB15k237Inductive_v1|B0|R3|Hits10|-0.002439|-0.014634|+0.009756|
|FB15k237Inductive_v1|B0|R4|MRR|-0.003439|-0.010717|+0.003831|
|FB15k237Inductive_v1|B0|R4|Hits1|-0.009756|-0.022012|+0.002439|
|FB15k237Inductive_v1|B0|R4|Hits3|+0.004878|+0.000000|+0.012195|
|FB15k237Inductive_v1|B0|R4|Hits10|+0.000000|-0.007317|+0.007317|
|FB15k237Inductive_v1|B0|R5|MRR|-0.003439|-0.010717|+0.003831|
|FB15k237Inductive_v1|B0|R5|Hits1|-0.009756|-0.022012|+0.002439|
|FB15k237Inductive_v1|B0|R5|Hits3|+0.004878|+0.000000|+0.012195|
|FB15k237Inductive_v1|B0|R5|Hits10|+0.000000|-0.007317|+0.007317|
|FB15k237Inductive_v1|T-4D|R1|MRR|-0.000381|-0.001028|+0.000178|
|FB15k237Inductive_v1|T-4D|R1|Hits1|+0.000000|+0.000000|+0.000000|
|FB15k237Inductive_v1|T-4D|R1|Hits3|-0.002439|-0.007317|+0.000000|
|FB15k237Inductive_v1|T-4D|R1|Hits10|+0.000000|+0.000000|+0.000000|
|FB15k237Inductive_v1|T-4D|R2|MRR|+0.003703|-0.003647|+0.011680|
|FB15k237Inductive_v1|T-4D|R2|Hits1|+0.007317|-0.004878|+0.019512|
|FB15k237Inductive_v1|T-4D|R2|Hits3|-0.002439|-0.017073|+0.012195|
|FB15k237Inductive_v1|T-4D|R2|Hits10|+0.002439|-0.009756|+0.014634|
|FB15k237Inductive_v1|T-4D|R3|MRR|+0.003360|-0.002672|+0.010223|
|FB15k237Inductive_v1|T-4D|R3|Hits1|+0.004878|-0.004878|+0.014634|
|FB15k237Inductive_v1|T-4D|R3|Hits3|+0.000000|-0.014634|+0.014634|
|FB15k237Inductive_v1|T-4D|R3|Hits10|+0.004878|-0.007317|+0.017073|
|FB15k237Inductive_v1|T-4D|R4|MRR|-0.003347|-0.018194|+0.013551|
|FB15k237Inductive_v1|T-4D|R4|Hits1|+0.007317|-0.014634|+0.031707|
|FB15k237Inductive_v1|T-4D|R4|Hits3|-0.019512|-0.046341|+0.004878|
|FB15k237Inductive_v1|T-4D|R4|Hits10|-0.009756|-0.026829|+0.007317|
|FB15k237Inductive_v1|T-4D|R5|MRR|-0.003347|-0.018194|+0.013551|
|FB15k237Inductive_v1|T-4D|R5|Hits1|+0.007317|-0.014634|+0.031707|
|FB15k237Inductive_v1|T-4D|R5|Hits3|-0.019512|-0.046341|+0.004878|
|FB15k237Inductive_v1|T-4D|R5|Hits10|-0.009756|-0.026829|+0.007317|
|WN18RRInductive_v1|B0|R1|MRR|+0.012697|+0.001112|+0.026924|
|WN18RRInductive_v1|B0|R1|Hits1|+0.018617|+0.000000|+0.039894|
|WN18RRInductive_v1|B0|R1|Hits3|-0.002660|-0.021277|+0.015957|
|WN18RRInductive_v1|B0|R1|Hits10|+0.018617|+0.005319|+0.034574|
|WN18RRInductive_v1|B0|R2|MRR|+0.007643|-0.001511|+0.018898|
|WN18RRInductive_v1|B0|R2|Hits1|+0.013298|-0.002660|+0.031915|
|WN18RRInductive_v1|B0|R2|Hits3|-0.005319|-0.018617|+0.007979|
|WN18RRInductive_v1|B0|R2|Hits10|+0.010638|+0.002660|+0.021277|
|WN18RRInductive_v1|B0|R3|MRR|+0.013868|+0.002414|+0.026706|
|WN18RRInductive_v1|B0|R3|Hits1|+0.021277|+0.002660|+0.042553|
|WN18RRInductive_v1|B0|R3|Hits3|-0.007979|-0.023936|+0.010638|
|WN18RRInductive_v1|B0|R3|Hits10|+0.015957|+0.002660|+0.031915|
|WN18RRInductive_v1|B0|R4|MRR|+0.008162|-0.000759|+0.020070|
|WN18RRInductive_v1|B0|R4|Hits1|+0.013298|-0.002660|+0.031915|
|WN18RRInductive_v1|B0|R4|Hits3|-0.002660|-0.015957|+0.010638|
|WN18RRInductive_v1|B0|R4|Hits10|+0.007979|+0.000000|+0.018617|
|WN18RRInductive_v1|B0|R5|MRR|+0.008162|-0.000759|+0.020070|
|WN18RRInductive_v1|B0|R5|Hits1|+0.013298|-0.002660|+0.031915|
|WN18RRInductive_v1|B0|R5|Hits3|-0.002660|-0.015957|+0.010638|
|WN18RRInductive_v1|B0|R5|Hits10|+0.007979|+0.000000|+0.018617|
|WN18RRInductive_v1|T-4D|R1|MRR|+0.016178|+0.002628|+0.031923|
|WN18RRInductive_v1|T-4D|R1|Hits1|+0.021277|+0.000000|+0.045213|
|WN18RRInductive_v1|T-4D|R1|Hits3|+0.007979|-0.007979|+0.026596|
|WN18RRInductive_v1|T-4D|R1|Hits10|+0.015957|+0.002660|+0.031915|
|WN18RRInductive_v1|T-4D|R2|MRR|+0.004984|-0.002991|+0.014529|
|WN18RRInductive_v1|T-4D|R2|Hits1|+0.005319|-0.007979|+0.021277|
|WN18RRInductive_v1|T-4D|R2|Hits3|+0.002660|-0.005319|+0.010638|
|WN18RRInductive_v1|T-4D|R2|Hits10|+0.005319|-0.005319|+0.015957|
|WN18RRInductive_v1|T-4D|R3|MRR|+0.016510|+0.005958|+0.028891|
|WN18RRInductive_v1|T-4D|R3|Hits1|+0.021277|+0.005319|+0.039894|
|WN18RRInductive_v1|T-4D|R3|Hits3|-0.002660|-0.015957|+0.010638|
|WN18RRInductive_v1|T-4D|R3|Hits10|+0.013298|+0.000000|+0.026596|
|WN18RRInductive_v1|T-4D|R4|MRR|+0.010967|-0.002387|+0.025160|
|WN18RRInductive_v1|T-4D|R4|Hits1|+0.010638|-0.010638|+0.031915|
|WN18RRInductive_v1|T-4D|R4|Hits3|+0.005319|-0.005319|+0.018617|
|WN18RRInductive_v1|T-4D|R4|Hits10|+0.013298|+0.000000|+0.026596|
|WN18RRInductive_v1|T-4D|R5|MRR|+0.014025|+0.000184|+0.028689|
|WN18RRInductive_v1|T-4D|R5|Hits1|+0.015957|-0.007979|+0.039894|
|WN18RRInductive_v1|T-4D|R5|Hits3|+0.007979|-0.005319|+0.021277|
|WN18RRInductive_v1|T-4D|R5|Hits10|+0.013298|+0.000000|+0.026596|
|NELLInductive_v1|B0|R1|MRR|+0.052948|+0.016701|+0.089288|
|NELLInductive_v1|B0|R1|Hits1|+0.070000|+0.025000|+0.115000|
|NELLInductive_v1|B0|R1|Hits3|+0.025000|-0.020000|+0.070000|
|NELLInductive_v1|B0|R1|Hits10|+0.045000|+0.020000|+0.075000|
|NELLInductive_v1|B0|R2|MRR|-0.037944|-0.081031|+0.005112|
|NELLInductive_v1|B0|R2|Hits1|-0.040000|-0.090000|+0.010000|
|NELLInductive_v1|B0|R2|Hits3|-0.045000|-0.095000|+0.005000|
|NELLInductive_v1|B0|R2|Hits10|-0.045000|-0.075000|-0.020000|
|NELLInductive_v1|B0|R3|MRR|+0.047750|+0.028167|+0.068833|
|NELLInductive_v1|B0|R3|Hits1|+0.055000|+0.025000|+0.085000|
|NELLInductive_v1|B0|R3|Hits3|+0.050000|+0.025000|+0.080000|
|NELLInductive_v1|B0|R3|Hits10|+0.000000|+0.000000|+0.000000|
|NELLInductive_v1|B0|R4|MRR|+0.061367|+0.028875|+0.094209|
|NELLInductive_v1|B0|R4|Hits1|+0.090000|+0.045000|+0.135000|
|NELLInductive_v1|B0|R4|Hits3|+0.040000|+0.005000|+0.080000|
|NELLInductive_v1|B0|R4|Hits10|-0.015000|-0.035000|+0.000000|
|NELLInductive_v1|B0|R5|MRR|+0.061436|+0.028728|+0.094786|
|NELLInductive_v1|B0|R5|Hits1|+0.090000|+0.045000|+0.135000|
|NELLInductive_v1|B0|R5|Hits3|+0.040000|+0.000000|+0.080000|
|NELLInductive_v1|B0|R5|Hits10|-0.015000|-0.035000|+0.000000|
|NELLInductive_v1|T-4D|R1|MRR|-0.005666|-0.038705|+0.028307|
|NELLInductive_v1|T-4D|R1|Hits1|-0.015000|-0.060000|+0.035000|
|NELLInductive_v1|T-4D|R1|Hits3|-0.020000|-0.055000|+0.015000|
|NELLInductive_v1|T-4D|R1|Hits10|+0.050000|+0.015000|+0.085000|
|NELLInductive_v1|T-4D|R2|MRR|+0.017988|-0.012079|+0.050463|
|NELLInductive_v1|T-4D|R2|Hits1|+0.025000|-0.015000|+0.065000|
|NELLInductive_v1|T-4D|R2|Hits3|-0.005000|-0.050000|+0.040000|
|NELLInductive_v1|T-4D|R2|Hits10|+0.025000|-0.005000|+0.060000|
|NELLInductive_v1|T-4D|R3|MRR|+0.057512|+0.029189|+0.086938|
|NELLInductive_v1|T-4D|R3|Hits1|+0.055000|+0.015000|+0.095125|
|NELLInductive_v1|T-4D|R3|Hits3|+0.075000|+0.035000|+0.115000|
|NELLInductive_v1|T-4D|R3|Hits10|+0.000000|-0.025000|+0.025000|
|NELLInductive_v1|T-4D|R4|MRR|-0.010614|-0.038384|+0.017435|
|NELLInductive_v1|T-4D|R4|Hits1|-0.045000|-0.090000|-0.004875|
|NELLInductive_v1|T-4D|R4|Hits3|+0.015000|-0.030000|+0.060000|
|NELLInductive_v1|T-4D|R4|Hits10|+0.010000|-0.020000|+0.040000|
|NELLInductive_v1|T-4D|R5|MRR|-0.010614|-0.038384|+0.017435|
|NELLInductive_v1|T-4D|R5|Hits1|-0.045000|-0.090000|-0.004875|
|NELLInductive_v1|T-4D|R5|Hits3|+0.015000|-0.030000|+0.060000|
|NELLInductive_v1|T-4D|R5|Hits10|+0.010000|-0.020000|+0.040000|
|FBIngram_25|B0|R1|MRR|+0.002997|+0.000818|+0.005236|
|FBIngram_25|B0|R1|Hits1|+0.003849|+0.000000|+0.007610|
|FBIngram_25|B0|R1|Hits3|+0.001662|-0.001924|+0.005511|
|FBIngram_25|B0|R1|Hits10|-0.002537|-0.004811|-0.000262|
|FBIngram_25|B0|R2|MRR|+0.002837|+0.000347|+0.005194|
|FBIngram_25|B0|R2|Hits1|+0.003237|-0.000962|+0.007348|
|FBIngram_25|B0|R2|Hits3|+0.001575|-0.002362|+0.005686|
|FBIngram_25|B0|R2|Hits10|-0.003761|-0.006211|-0.001310|
|FBIngram_25|B0|R3|MRR|+0.002287|-0.000036|+0.004520|
|FBIngram_25|B0|R3|Hits1|+0.002974|-0.000875|+0.006648|
|FBIngram_25|B0|R3|Hits3|+0.001225|-0.002537|+0.005161|
|FBIngram_25|B0|R3|Hits10|-0.002537|-0.004374|-0.000875|
|FBIngram_25|B0|R4|MRR|+0.001928|-0.000693|+0.004442|
|FBIngram_25|B0|R4|Hits1|+0.002624|-0.001489|+0.006910|
|FBIngram_25|B0|R4|Hits3|+0.000875|-0.003324|+0.005161|
|FBIngram_25|B0|R4|Hits10|-0.005073|-0.007698|-0.002449|
|FBIngram_25|T-4D|R1|MRR|+0.001941|+0.000370|+0.003549|
|FBIngram_25|T-4D|R1|Hits1|+0.001837|-0.001050|+0.004549|
|FBIngram_25|T-4D|R1|Hits3|+0.002712|-0.000002|+0.005426|
|FBIngram_25|T-4D|R1|Hits10|-0.001312|-0.002799|+0.000175|
|FBIngram_25|T-4D|R2|MRR|+0.000051|-0.002501|+0.002629|
|FBIngram_25|T-4D|R2|Hits1|-0.000875|-0.005161|+0.003324|
|FBIngram_25|T-4D|R2|Hits3|+0.001225|-0.002974|+0.005598|
|FBIngram_25|T-4D|R2|Hits10|-0.002012|-0.004636|+0.000612|
|FBIngram_25|T-4D|R3|MRR|-0.001173|-0.003503|+0.001176|
|FBIngram_25|T-4D|R3|Hits1|-0.001662|-0.005598|+0.002099|
|FBIngram_25|T-4D|R3|Hits3|-0.002187|-0.006123|+0.001837|
|FBIngram_25|T-4D|R3|Hits10|-0.001837|-0.004549|+0.000700|
|FBIngram_25|T-4D|R4|MRR|+0.002303|+0.000355|+0.004218|
|FBIngram_25|T-4D|R4|Hits1|+0.002712|-0.000612|+0.005948|
|FBIngram_25|T-4D|R4|Hits3|+0.000087|-0.002974|+0.003149|
|FBIngram_25|T-4D|R4|Hits10|-0.002712|-0.004986|-0.000350|

## Top 11 至 30 扩召回收益

|图谱|基线|方法|原 Filtered 11–30 查询|进入前十|原前十跌出|Hits@10 净变化|
|---|---|---|---:|---:|---:|---:|
|FB15k237Inductive_v1|B0|R1|24|0|0|+0.000000|
|FB15k237Inductive_v1|B0|R2|24|2|7|-0.012195|
|FB15k237Inductive_v1|B0|R3|24|3|4|-0.002439|
|FB15k237Inductive_v1|B0|R4|24|1|1|+0.000000|
|FB15k237Inductive_v1|B0|R5|24|1|1|+0.000000|
|FB15k237Inductive_v1|T-4D|R1|25|0|0|+0.000000|
|FB15k237Inductive_v1|T-4D|R2|25|4|3|+0.002439|
|FB15k237Inductive_v1|T-4D|R3|25|4|2|+0.004878|
|FB15k237Inductive_v1|T-4D|R4|25|5|9|-0.009756|
|FB15k237Inductive_v1|T-4D|R5|25|5|9|-0.009756|
|WN18RRInductive_v1|B0|R1|30|8|1|+0.018617|
|WN18RRInductive_v1|B0|R2|30|4|0|+0.010638|
|WN18RRInductive_v1|B0|R3|30|7|1|+0.015957|
|WN18RRInductive_v1|B0|R4|30|3|0|+0.007979|
|WN18RRInductive_v1|B0|R5|30|3|0|+0.007979|
|WN18RRInductive_v1|T-4D|R1|26|7|1|+0.015957|
|WN18RRInductive_v1|T-4D|R2|26|3|1|+0.005319|
|WN18RRInductive_v1|T-4D|R3|26|6|1|+0.013298|
|WN18RRInductive_v1|T-4D|R4|26|6|1|+0.013298|
|WN18RRInductive_v1|T-4D|R5|26|6|1|+0.013298|
|NELLInductive_v1|B0|R1|26|9|0|+0.045000|
|NELLInductive_v1|B0|R2|26|0|9|-0.045000|
|NELLInductive_v1|B0|R3|26|0|0|+0.000000|
|NELLInductive_v1|B0|R4|26|0|3|-0.015000|
|NELLInductive_v1|B0|R5|26|0|3|-0.015000|
|NELLInductive_v1|T-4D|R1|39|13|3|+0.050000|
|NELLInductive_v1|T-4D|R2|39|8|3|+0.025000|
|NELLInductive_v1|T-4D|R3|39|3|3|+0.000000|
|NELLInductive_v1|T-4D|R4|39|6|4|+0.010000|
|NELLInductive_v1|T-4D|R5|39|6|4|+0.010000|
|FBIngram_25|B0|R1|1513|73|102|-0.002537|
|FBIngram_25|B0|R2|1513|79|122|-0.003761|
|FBIngram_25|B0|R3|1513|33|62|-0.002537|
|FBIngram_25|B0|R4|1513|85|143|-0.005073|
|FBIngram_25|T-4D|R1|1517|29|44|-0.001312|
|FBIngram_25|T-4D|R2|1517|104|127|-0.002012|
|FBIngram_25|T-4D|R3|1517|103|124|-0.001837|
|FBIngram_25|T-4D|R4|1517|74|105|-0.002712|

候选池的排名与全图 Filtered 排名不同，可能存在原 Filtered 排名≤10但候选池排名>10的目标。主表 Hits@10 差值由完整排名直接计算，不能只用“11–30 提拔次数减去跌出次数”替代。

## C D E3a F 的结构排序诊断

|图谱|基线|方法|Top10 条件结构 MRR|池内11–30答案|严格超过Top10负例|
|---|---|---|---:|---:|---:|
|FB15k237Inductive_v1|B0|R1|0.28982178774815315|24|0|
|FB15k237Inductive_v1|B0|R2|0.550154137568386|24|3|
|FB15k237Inductive_v1|B0|R3|0.6130992996048009|24|3|
|FB15k237Inductive_v1|B0|R4|0.45770283755366287|24|2|
|FB15k237Inductive_v1|T-4D|R1|0.289071345027757|25|0|
|FB15k237Inductive_v1|T-4D|R2|0.5424820622651216|25|3|
|FB15k237Inductive_v1|T-4D|R3|0.6113103599508593|25|4|
|FB15k237Inductive_v1|T-4D|R4|0.4560915912840876|25|2|
|WN18RRInductive_v1|B0|R1|0.2403592445062503|29|0|
|WN18RRInductive_v1|B0|R2|0.4348226737972661|29|5|
|WN18RRInductive_v1|B0|R3|0.7789848362910708|29|8|
|WN18RRInductive_v1|B0|R4|0.3986664545375982|29|2|
|WN18RRInductive_v1|T-4D|R1|0.23861569692078194|26|0|
|WN18RRInductive_v1|T-4D|R2|0.43383261747668583|26|4|
|WN18RRInductive_v1|T-4D|R3|0.770557637222642|26|10|
|WN18RRInductive_v1|T-4D|R4|0.3961665076919317|26|2|
|NELLInductive_v1|B0|R1|0.2028164860281649|55|0|
|NELLInductive_v1|B0|R2|0.6518911745189127|55|0|
|NELLInductive_v1|B0|R3|0.7799325370493259|55|0|
|NELLInductive_v1|B0|R4|0.45113718398389957|55|8|
|NELLInductive_v1|T-4D|R1|0.20485154576063658|57|0|
|NELLInductive_v1|T-4D|R2|0.599510254055709|57|0|
|NELLInductive_v1|T-4D|R3|0.7324016681971232|57|0|
|NELLInductive_v1|T-4D|R4|0.37300836542146726|57|8|
|FBIngram_25|B0|R1|0.19234137408638538|1691|1|
|FBIngram_25|B0|R2|0.39393008612014346|1691|79|
|FBIngram_25|B0|R3|0.4042373324975603|1691|47|
|FBIngram_25|B0|R4|0.36160044673267505|1691|104|
|FBIngram_25|T-4D|R1|0.1924712603164217|1692|1|
|FBIngram_25|T-4D|R2|0.3937893885957841|1692|83|
|FBIngram_25|T-4D|R3|0.40538567778131185|1692|60|
|FBIngram_25|T-4D|R4|0.3603992329263392|1692|108|

这些诊断使用各自新候选池和官方测试集，与历史未过滤候选、合并 valid+test 的条件指标存在口径差异。它们衡量结构证据排序，不是全图 Filtered MRR。

## LLM 触发与正负翻转

|图谱|基线|触发查询|触发率|真实响应数|拒答|拒答率|格式回退|正翻转|负翻转|总Token|平均延迟秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FB15k237Inductive_v1|B0|191|46.59%|191|52|27.23%|0|0|0|773976|1.433|
|FB15k237Inductive_v1|T-4D|194|47.32%|194|56|28.87%|0|0|0|804444|1.431|
|WN18RRInductive_v1|B0|131|34.84%|131|12|9.16%|0|0|0|518492|1.523|
|WN18RRInductive_v1|T-4D|129|34.31%|129|11|8.53%|0|2|0|497683|1.505|
|NELLInductive_v1|B0|120|60.00%|120|59|49.17%|0|0|0|684914|1.508|
|NELLInductive_v1|T-4D|120|60.00%|120|42|35.00%|0|0|0|680597|1.402|
|FBIngram_25|B0|跳过||||||||
|FBIngram_25|T-4D|跳过||||||||

Token 与延迟按当前运行使用的真实响应汇总；命中历史响应缓存不增加网络请求。校准和连通性调用单独记录，未混入测试调用计数。

## 实际图算子核验与内存峰值

- FB15k237Inductive_v1/B0：核验 176 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.239338；发送端行常数抵消最大误差为 4.44e-16。
  进程峰值 RSS：0.866 GB，限制为 1.800 GB。
- FB15k237Inductive_v1/T-4D：核验 176 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.239338；发送端行常数抵消最大误差为 4.44e-16。
  进程峰值 RSS：0.763 GB，限制为 1.800 GB。
- WN18RRInductive_v1/B0：核验 14 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.141725；发送端行常数抵消最大误差为 2.22e-16。
  进程峰值 RSS：0.821 GB，限制为 1.800 GB。
- WN18RRInductive_v1/T-4D：核验 14 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.141725；发送端行常数抵消最大误差为 2.22e-16。
  进程峰值 RSS：0.739 GB，限制为 1.800 GB。
- NELLInductive_v1/B0：核验 14 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.272111；发送端行常数抵消最大误差为 2.22e-16。
  进程峰值 RSS：0.862 GB，限制为 1.800 GB。
- NELLInductive_v1/T-4D：核验 14 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.272111；发送端行常数抵消最大误差为 2.22e-16。
  进程峰值 RSS：0.740 GB，限制为 1.800 GB。
- FBIngram_25/B0：核验 394 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.394308；发送端行常数抵消最大误差为 6.66e-16。
  进程峰值 RSS：1.132 GB，限制为 1.800 GB。
- FBIngram_25/T-4D：核验 394 个查询关系，全部前后向行归一化断言通过；与 E3a 的前向概率最大差异为 0.394308；发送端行常数抵消最大误差为 6.66e-16。
  进程峰值 RSS：1.055 GB，限制为 1.800 GB。

## 候选协议差异与数据完整性

- FB15k237Inductive_v1 / B0：41/410 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 44 个相应候选。
- FB15k237Inductive_v1 / T-4D：42/410 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 45 个相应候选。
- WN18RRInductive_v1 / B0：44/376 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 45 个相应候选。
- WN18RRInductive_v1 / T-4D：45/376 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 46 个相应候选。
- NELLInductive_v1 / B0：99/200 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 1265 个相应候选。
- NELLInductive_v1 / T-4D：99/200 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 1187 个相应候选。
- FBIngram_25 / B0：5922/11432 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 21324 个相应候选。
- FBIngram_25 / T-4D：5912/11432 条查询的候选池与标准先 Filtered 后取 Top-30 不同，合计缺少 21210 个相应候选。

## 真实案例与文本覆盖

FB15k237Inductive_v1 / B0：本地文本映射覆盖 0/1093。缺失时保留原始 ID 或 NELL 概念前缀，不编造实体名。

- FB15k237Inductive_v1/B0/1_0：结构排名 2 → 最终排名 2；abstain=False；模型说明：候选87和99均通过从头部出发的提名/获奖路径可达，路径逻辑与查询关系一致；候选854的路径涉及电影表演关系，与查询的奖项提名关系无关，故不纳入。。完整原始路径及边权见 `FB15k237Inductive_v1/B0/cases/1_0.json`。

- FB15k237Inductive_v1/B0/2_1：结构排名 19 → 最终排名 19；abstain=False；模型说明：查询要求通过逆关系预测原头实体，候选506、713、305的路径均以逆关系步骤从/m/03shpq出发，经演员或奖项链最终以逆关系指向候选，逻辑上支持候选作为原头实体，其中506路径最短且权重最高。。完整原始路径及边权见 `FB15k237Inductive_v1/B0/cases/2_1.json`。

- FB15k237Inductive_v1/B0/3_1：结构排名 3 → 最终排名 3；abstain=False；模型说明：候选110的路径经/m/01_d4和/m/04htfd，最后一步正是查询的逆关系（公司→职位任职者），逻辑上直接支持；候选234路径末步为政府职位管辖关系，与查询无关；其余候选仅经出生地关系，不足以支持。。完整原始路径及边权见 `FB15k237Inductive_v1/B0/cases/3_1.json`。

- FB15k237Inductive_v1/B0/4_1：结构排名 1 → 最终排名 1；abstain=False；模型说明：候选1通过一条直接路径（/m/02h9_l 经 award award nominee award nominations 关系指向 /m/01c99j）与查询头相连，该路径与逆关系查询方向一致，支持度最高。。完整原始路径及边权见 `FB15k237Inductive_v1/B0/cases/4_1.json`。
FB15k237Inductive_v1 / T-4D：本地文本映射覆盖 0/1093。缺失时保留原始 ID 或 NELL 概念前缀，不编造实体名。

- FB15k237Inductive_v1/T-4D/1_0：结构排名 2 → 最终排名 2；abstain=False；模型说明：两条候选路径均通过提名/获奖关系链从头部实体连到候选，路径逻辑一致且无冲突，但路径权重接近，无法唯一确定。。完整原始路径及边权见 `FB15k237Inductive_v1/T-4D/cases/1_0.json`。

- FB15k237Inductive_v1/T-4D/3_0：结构排名 388 → 最终排名 388；abstain=False；模型说明：候选272、380、111均通过同一三跳路径从查询头经职位/头衔节点连接到目标，且末跳正是所查询的‘business job title people with this title. business employment tenure company’关系，路径逻辑支持预测；其余候选路径仅涉及政府职位或管辖关系，与查询关系无关。。完整原始路径及边权见 `FB15k237Inductive_v1/T-4D/cases/3_0.json`。

- FB15k237Inductive_v1/T-4D/3_1：结构排名 6 → 最终排名 6；abstain=True；模型说明：查询要求通过逆关系预测与职位任职公司相关的实体，但所有候选路径均从地点包含、出生地、相邻或居住关系出发，未涉及任何职位或雇佣关系，因此证据不足以支持判断。。完整原始路径及边权见 `FB15k237Inductive_v1/T-4D/cases/3_1.json`。

- FB15k237Inductive_v1/T-4D/4_0：结构排名 97 → 最终排名 97；abstain=False；模型说明：候选180通过同族群节点经影响同伴关系直接相连，路径逻辑最贴合查询；候选364和208的路径需经电影或奖项等中间关系，仅属普通相关性，支持较弱。。完整原始路径及边权见 `FB15k237Inductive_v1/T-4D/cases/4_0.json`。
WN18RRInductive_v1 / B0：本地文本映射覆盖 0/922。缺失时保留原始 ID 或 NELL 概念前缀，不编造实体名。

- WN18RRInductive_v1/B0/0_0：结构排名 105 → 最终排名 105；abstain=False；模型说明：候选653、699、234均与头实体存在直接的derivationally related form或逆关系路径，这类形态派生关系是similar to查询可依据的最直接证据，而其余候选仅通过hypernym或间接派生链连接，逻辑蕴含较弱。。完整原始路径及边权见 `WN18RRInductive_v1/B0/cases/0_0.json`。

- WN18RRInductive_v1/B0/0_1：结构排名 11 → 最终排名 11；abstain=False；模型说明：查询是逆相似关系，即寻找与00444519相似的节点；候选509通过一条直接的similar to路径从00444519指向它，符合逆关系语义，而其他候选均需经过额外关系步骤，逻辑蕴含较弱。。完整原始路径及边权见 `WN18RRInductive_v1/B0/cases/0_1.json`。

- WN18RRInductive_v1/B0/12_0：结构排名 2 → 最终排名 2；abstain=False；模型说明：候选127和101均通过标注的[Inverse of derivationally related form]逆关系路径与查询头直接相连，符合查询所要求的派生相关形式关系预测。。完整原始路径及边权见 `WN18RRInductive_v1/B0/cases/12_0.json`。

- WN18RRInductive_v1/B0/12_1：结构排名 1 → 最终排名 1；abstain=False；模型说明：查询是逆派生关系，即预测原派生关系的头；候选42有直接路径 07976936 -[derivationally related form]-> 01292885，正好满足逆关系所要求的原头预测，且路径权重最高。。完整原始路径及边权见 `WN18RRInductive_v1/B0/cases/12_1.json`。
WN18RRInductive_v1 / T-4D：本地文本映射覆盖 0/922。缺失时保留原始 ID 或 NELL 概念前缀，不编造实体名。

- WN18RRInductive_v1/T-4D/0_0：结构排名 11 → 最终排名 11；abstain=False；模型说明：候选573通过一条直接路径与查询头相连：00445169 的 [Inverse of similar to] 指向 00444519，逆关系解释后即表示 00445169 与 00444519 相似，直接支持查询关系；其余候选仅通过派生相关或上下位关系间接相连，不能逻辑蕴含相似关系。。完整原始路径及边权见 `WN18RRInductive_v1/T-4D/cases/0_0.json`。

- WN18RRInductive_v1/T-4D/0_1：结构排名 11 → 最终排名 11；abstain=False；模型说明：查询要求逆关系[Inverse of similar to]的尾部，即寻找原头00444519的相似词；候选509通过直接路径00444519 similar to 00445169给出，正是该逆关系对应的原头预测，其余候选均需经派生或上位等多跳关系，逻辑蕴含较弱。。完整原始路径及边权见 `WN18RRInductive_v1/T-4D/cases/0_1.json`。

- WN18RRInductive_v1/T-4D/12_0：结构排名 2 → 最终排名 2；abstain=False；模型说明：候选127和101均通过一条标注为[Inverse of derivationally related form]的路径直接连回查询头01292885，符合查询所问的派生相关词关系（逆关系按标注解释为原头预测），故支持；其余候选路径含多跳或verb group/hypernym等非目标关系，逻辑蕴含较弱。。完整原始路径及边权见 `WN18RRInductive_v1/T-4D/cases/12_0.json`。

- WN18RRInductive_v1/T-4D/12_1：结构排名 1 → 最终排名 1；abstain=False；模型说明：查询是逆派生关系预测原头，候选42有直接派生关系路径，且经01428853、07988857的派生链也支持；472和266通过派生或动词组路径间接关联，但证据弱于42。。完整原始路径及边权见 `WN18RRInductive_v1/T-4D/cases/12_1.json`。
NELLInductive_v1 / B0：本地文本映射覆盖 0/225。缺失时保留原始 ID 或 NELL 概念前缀，不编造实体名。

- NELLInductive_v1/B0/0_1：结构排名 3 → 最终排名 3；abstain=True；模型说明：查询要求逆关系 concept:subpartof 的尾部，但所有候选路径均通过 acquired、affiliatedwith、belongstoorganization、controls 或 collaborateswith 等关系连接，没有一条路径包含 subpartof 或其逆关系，因此证据不足以判断。。完整原始路径及边权见 `NELLInductive_v1/B0/cases/0_1.json`。

- NELLInductive_v1/B0/1_1：结构排名 1 → 最终排名 1；abstain=False；模型说明：查询要求逆关系 [Inverse of concept:agentbelongstoorganization] 的原头预测，即寻找属于 PBS 的代理；候选 213 有直接路径 concept:company:pbs concept:agentcontrols concept:televisionstation:wfiq，且另有 [Inverse of concept:subpartof] 路径，逻辑上支持该台受 PBS 控制/归属，其余候选仅通过附属或合作等间接路径，不足以支持归属关系。。完整原始路径及边权见 `NELLInductive_v1/B0/cases/1_1.json`。

- NELLInductive_v1/B0/2_1：结构排名 11 → 最终排名 11；abstain=True；模型说明：查询关系为 agentcollaborateswithagent 的逆关系，但所有候选路径均经由 acquired、affiliatedwith、subpartof 等关系连接电视台，未出现任何 agentcollaborateswithagent 关系，路径与查询无关，无法判断。。完整原始路径及边权见 `NELLInductive_v1/B0/cases/2_1.json`。

- NELLInductive_v1/B0/37_1：结构排名 9 → 最终排名 3；abstain=False；模型说明：查询要求逆关系 [Inverse of concept:agentbelongstoorganization] 的原头预测，即寻找属于 PBS 的代理；候选 213 有直接路径 concept:company:pbs concept:agentcontrols concept:televisionstation:wfiq，且另有 [Inverse of concept:subpartof] 路径，逻辑上支持该台受 PBS 控制/归属，其余候选仅通过附属或合作等间接路径，不足以支持归属关系。。完整原始路径及边权见 `NELLInductive_v1/B0/cases/37_1.json`。
NELLInductive_v1 / T-4D：本地文本映射覆盖 0/225。缺失时保留原始 ID 或 NELL 概念前缀，不编造实体名。

- NELLInductive_v1/T-4D/0_1：结构排名 2 → 最终排名 2；abstain=False；模型说明：查询为逆关系 [Inverse of concept:subpartof]，即预测原头实体；候选 174、212、172 均通过 [Inverse of concept:agentbelongstoorganization] 或 concept:agentcontrols 等直接路径与 company:pbs 相连，其中 174 的路径权重最高且同社区，故支持度最强。。完整原始路径及边权见 `NELLInductive_v1/T-4D/cases/0_1.json`。

- NELLInductive_v1/T-4D/1_1：结构排名 7 → 最终排名 7；abstain=False；模型说明：查询为逆关系，需找指向 company:pbs 的 agentbelongstoorganization 边；候选 211 和 198 均存在直接路径 [Inverse of concept:subpartof] 从 company:pbs 指向自身，按逆关系解释即候选属于 pbs，且路径权重最高，其余候选仅通过电视台网络间接关联，支持较弱。。完整原始路径及边权见 `NELLInductive_v1/T-4D/cases/1_1.json`。

- NELLInductive_v1/T-4D/2_1：结构排名 17 → 最终排名 17；abstain=True；模型说明：所有候选路径均通过 PBS 的收购、附属或子部分关系连接到电视台，未提供任何关于 agentcollaborateswithagent 逆关系（即谁与 PBS 协作）的证据，因此不足以判断。。完整原始路径及边权见 `NELLInductive_v1/T-4D/cases/2_1.json`。

- NELLInductive_v1/T-4D/3_0：结构排名 2 → 最终排名 2；abstain=True；模型说明：所有候选路径均未包含 concept:agentcontrols 关系，仅涉及收购、附属、雇佣等普通关联，无法逻辑蕴含查询关系。。完整原始路径及边权见 `NELLInductive_v1/T-4D/cases/3_0.json`。

## 结论与限制

- FB15k237Inductive_v1 / B0：R4 相对 M0 的 MRR 差值为 -0.003439；R4 点估计降低。显著性以配对区间为准。
- FB15k237Inductive_v1 / T-4D：R4 相对 M0 的 MRR 差值为 -0.003347；R4 点估计降低。显著性以配对区间为准。
- WN18RRInductive_v1 / B0：R4 相对 M0 的 MRR 差值为 +0.008162；R4 点估计提高。显著性以配对区间为准。
- WN18RRInductive_v1 / T-4D：R4 相对 M0 的 MRR 差值为 +0.010967；R4 点估计提高。显著性以配对区间为准。
- NELLInductive_v1 / B0：R4 相对 M0 的 MRR 差值为 +0.061367；R4 点估计提高。显著性以配对区间为准。
- NELLInductive_v1 / T-4D：R4 相对 M0 的 MRR 差值为 -0.010614；R4 点估计降低。显著性以配对区间为准。
- FBIngram_25 / B0：R4 相对 M0 的 MRR 差值为 +0.001928；R4 点估计提高。显著性以配对区间为准。
- FBIngram_25 / T-4D：R4 相对 M0 的 MRR 差值为 +0.002303；R4 点估计提高。显著性以配对区间为准。

目标图谱已有历史研究暴露；本次新验证集选择是目标域开发协议，不能称为无目标选参的 P0 或首次盲测。保护机制不保证零退化。只运行一个固定模型种子；置信区间描述查询抽样不确定性，不涵盖训练种子和 LLM 服务版本变化。

API 参数依据 [DeepSeek 官方 Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)；模型为 deepseek-flash，关闭 thinking，要求 JSON 输出。API 凭证只从环境或原始本地文档读入进程，不写入源码、报告或运行配置。
