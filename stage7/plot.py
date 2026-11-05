"""Standalone scientific figures, built only from final inspectable CSVs."""
import csv
from collections import defaultdict
import numpy as np
from stage7.util import ROOT,read,log,dump,sha


def table(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':9.5,
         'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'pdf.fonttype':42,'svg.fonttype':'none'})
    font=font_manager.findfont('Microsoft YaHei',fallback_to_default=False)
    dump(ROOT/'results/figure_environment.json',dict(matplotlib=matplotlib.__version__,backend=matplotlib.get_backend(),
        font=font,font_sha256=sha(font),default_font_size_pt=9.5,minimum_annotation_font_size_pt=8.5,
        png_dpi=180,pdf_fonttype=42,svg_text=True,vector_formats=['pdf','svg']))
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    def save(fig,name):
        for fmt in ['png','pdf','svg']:fig.savefig(dest/(name+'.'+fmt),bbox_inches='tight')
        plt.close(fig)
    def ink(value,mapper):
        rgb=np.asarray(mapper.cmap(mapper.norm(value)))[:3]
        linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        luminance=float(linear@np.asarray([.2126,.7152,.0722]))
        return 'black' if luminance>.179 else 'white'
    p=read(ROOT/'protocol.json')
    labels={k:k.replace('Inductive','').replace('_external_new','').replace('_new','').replace('ConceptNet100k_external','ConceptNet') for _,k in p['targets']}
    labels['WikiTopicsMT4_health_new']='WikiMT4-health'
    preds=table('proxy_formal_prediction.csv');corr=table('correlations.csv')
    fig,axs=plt.subplots(2,2,figsize=(7,6.2),constrained_layout=True)
    choices=[('stage6_legacy','capped15%'),('A','0.1'),('equal_mean','all'),('worst_mean','all')]
    for panel,(ax,(est,den)) in enumerate(zip(axs.ravel(),choices)):
        rows=[r for r in preds if r['alpha']=='0.25' and r['estimator']==est and r['density']==den]
        for r in rows:
            new=r['group']=='new_prospective'
            ax.scatter(float(r['proxy_gain']),float(r['formal_gain']),s=43,marker='^' if new else 'o',
                color='#D55E00' if new else '#0072B2',edgecolor='white',linewidth=.5,zorder=3)
            if new or r['key']=='ConceptNet100k_external':
                offsets={'UMLS_external_new':(-40,6),'ConceptNet100k_external':(15,25),
                         'HM_1k_new':(10,16) if panel==3 else (-38,16),
                         'WikiTopicsMT4_health_new':(12,-20)}
                ax.annotate(labels[r['key']],(float(r['proxy_gain']),float(r['formal_gain'])),
                    xytext=offsets[r['key']],textcoords='offset points',fontsize=8.5,
                    arrowprops=dict(arrowstyle='-',lw=.6,color='.4'),
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.5))
        cs=[r for r in corr if r['scope']=='all' and r['estimator']==est and r['density']==den]
        if cs:
            c=cs[0]
            ax.text(.03,.97,f"n={c['n']}\nr={float(c['pearson']):.3f}，ρ={float(c['spearman']):.3f}" if c['pearson'] else f"n={c['n']}，相关性未定义",transform=ax.transAxes,va='top',fontsize=8.5)
        ax.axhline(0,color='.7',lw=.7);ax.axvline(0,color='.7',lw=.7)
        titles={'stage6_legacy':'旧采样（256上限）','A':'A自然频率（10%）',
                'equal_mean':'12环境等权均值','worst_mean':'12环境最坏均值'}
        ax.set(title=titles[est],xlabel='代理选择集配对MRR增益',ylabel='正式任务配对MRR增益')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker='o',color='#0072B2',linestyle='',label='历史暴露图'),
             Line2D([],[],marker='^',color='#D55E00',linestyle='',label='新增结果图（来源相关性另述）')]
    fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.05),fontsize=8.5)
    save(fig,'proxy_formal_scatter')
    dist=table('proxy_formal_distribution.csv')
    keys=[k for _,k in p['targets'] if any(r['key']==k for r in dist)]
    envs=['stage6_legacy','A','B','C','D']
    matrix=np.array([[np.mean([float(r['js_to_formal']) for r in dist if r['key']==k and r['distribution']==e and r['js_to_formal']]) for e in envs] for k in keys])
    fig,ax=plt.subplots(figsize=(7, max(4.5,len(keys)*.32)),constrained_layout=True)
    im=ax.pcolormesh(np.arange(6)-.5,np.arange(len(keys)+1)-.5,matrix,vmin=0,vmax=.85,cmap='cividis',shading='flat')
    ax.invert_yaxis()
    ax.set_xticks(range(5),['阶段六原采样','A 自然','B 关系均衡','C 度数','D 模体'])
    ax.set_yticks(range(len(keys)),[labels[k] for k in keys]);ax.set_title('代理与正式查询关系频率的Jensen–Shannon距离')
    for i in range(len(keys)):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color=ink(matrix[i,j],im))
    cb=fig.colorbar(im,ax=ax,label='JS距离（越低越接近）',shrink=.75)
    cb.solids.set_rasterized(False)
    save(fig,'relation_distribution_shift')
    proxy=table('proxy_metrics.csv')
    fig,axs=plt.subplots(2,2,figsize=(7,max(7,len(keys)*.47)),constrained_layout=True,sharey=True)
    values=[]
    for env in 'ABCD':
        values.append(np.array([[np.mean([float(r['gain']) for r in proxy if r['key']==k and r['environment']==env and r['density']==rate and r['alpha']=='0.25']) for rate in ['0.05','0.1','0.15']] for k in keys]))
    limit=max(.001,max(float(abs(v).max()) for v in values))
    for ax,env,v in zip(axs.ravel(),'ABCD',values):
        im=ax.pcolormesh(np.arange(4)-.5,np.arange(len(keys)+1)-.5,v,cmap='RdBu_r',
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),shading='flat')
        ax.set_xticks([0,1,2],['5%','10%','15%']);ax.set_title('环境 '+env)
        ax.set_yticks(range(len(keys)),[labels[k] for k in keys])
        for i in range(len(keys)):
            for j in range(3):ax.text(j,i,f'{v[i,j]:+.3f}',ha='center',va='center',fontsize=8.5,color=ink(v[i,j],im))
    axs[0,0].invert_yaxis()
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),label='α=0.25相对α0的代理MRR增益',shrink=.75)
    cb.solids.set_rasterized(False)
    save(fig,'density_environment_gain')
    selection=table('per_seed_metrics.csv')
    colors=['#b9bec4','#376c95','#2c8b7d','#ddaa45','#b64a39']
    fig,axs=plt.subplots(1,3,figsize=(9,max(5,len(keys)*.35)),constrained_layout=True,sharey=True)
    for ax,method in zip(axs,['stage6_lcb','single_lcb','robust_lcb']):
        left=np.zeros(len(keys))
        for a,col,hatch in zip([0,.1,.25,.5,1],colors,['','..','//','xx','\\\\']):
            fractions=[]
            for key in keys:
                rows=[r for r in selection if r['key']==key and r['method']==method]
                fractions.append(sum(float(r['alpha'])==a for r in rows)/max(len(rows),1))
            ax.barh(range(len(keys)),fractions,left=left,color=col,label=str(a),height=.72,hatch=hatch,edgecolor='white',linewidth=.3)
            left+=np.asarray(fractions)
        ax.set(yticks=range(len(keys)),yticklabels=[labels[k] for k in keys],xlim=(0,1),title=method,xlabel='最终α选择比例（共9次）')
        ax.tick_params(labelsize=11);ax.xaxis.label.set_size(11);ax.title.set_size(12)
        ax.invert_yaxis()
    axs[-1].legend(title='α',loc='upper left',bbox_to_anchor=(1,1),fontsize=11,title_fontsize=11)
    save(fig,'selection_stability')
    comparisons=table('paired_bootstrap.csv')
    rows=[r for r in comparisons if r['method']=='robust_lcb' and r['reference']=='stage6_lcb']
    fig,ax=plt.subplots(figsize=(7,max(4.5,len(rows)*.32)),constrained_layout=True)
    for i,r in enumerate(rows):
        value,lo,hi=map(float,[r['delta'],r['lo'],r['hi']])
        ax.errorbar(value,i,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='^' if r['group']=='new_prospective' else 'o',color='#D55E00' if r['group']=='new_prospective' else '#0072B2',capsize=2)
    ax.set_yticks(range(len(rows)),[labels[r['key']] for r in rows]);ax.invert_yaxis();ax.axvline(0,color='.5',lw=.8)
    ax.set(title='分布鲁棒LCB相对阶段六单环境LCB：逐事实配对95%区间',xlabel='MRR差（3训练种子平均，代理种子6101）')
    ax.legend(handles=handles,fontsize=8.5,loc='lower center',bbox_to_anchor=(.5,1.09),ncol=2)
    save(fig,'robust_vs_single_forest')
    def coeff(est,name):
        matches=[r for r in corr if r['scope']=='all' and r['estimator']==est and r['alpha']=='0.25']
        return f"{float(matches[0][name]):.3f}" if matches and matches[0][name] else '未定义'
    a10=next(r for r in corr if r['scope']=='all' and r['estimator']=='A' and r['density']=='0.1')
    without_umls=next(r for r in corr if r['scope']=='all_without_UMLS' and r['estimator']=='equal_mean')
    density_tests=table('density_sensitivity.csv')
    significant_density=sum(float(r['holm_p'])<.05 for r in density_tests)
    flips=len({keys[i] for v in values for i in range(len(keys)) if v[i].min()<0<v[i].max()})
    robust_rows=[r for r in selection if r['method']=='robust_lcb']
    zero=sum(float(r['alpha'])==0 for r in robust_rows)/max(1,len(robust_rows))
    positives=sum(float(r['delta'])>0 for r in rows);negatives=sum(float(r['delta'])<0 for r in rows)
    captions=f'''# 图注与统计口径

**图1：代理与正式收益。** 等权多环境估计的图级Spearman为{coeff('equal_mean','spearman')}，原采样为{coeff('stage6_legacy','spearman')}，新单环境A/10%为{float(a10['spearman']):.3f}；不能据此宣布多环境优于单环境。剔除UMLS后等权多环境Pearson为{float(without_umls['pearson']):.3f}，显示高收益点的影响；完整逐图留一见correlation_influence.csv。所有点使用固定α0.25相对同检查点α0的双向filtered MRR增益，先在图内平均训练/代理划分，相关性以图为单位。r为Pearson、ρ为Spearman；圆点是历史暴露图，三角是新增结果图，来源相关性和有效样本数另见正文。旧采样的capped15%只是名义率，实际掩码率并不相同。

**图2：关系分布偏移。** 不同代理环境与正式查询的关系分布距离各异，精确数值标在每个单元格。JS为采用自然对数的Jensen–Shannon距离，越低越接近；对三个代理种子取均值。该分析仅在正式选择封存后读取查询标签，不进入选择器。

**图3：密度敏感性。** 有{flips}图的平均代理增益在至少一个环境内随名义密度改变符号，但{len(density_tests)}项配对密度检验中Holm<0.05的为{significant_density}项。每格是α0.25相对α0的选择集MRR点差，对三个既有训练种子和三个代理划分取均值；检验使用主代理种子6101且先平均训练种子，两者统计口径不同。A自然、B关系均衡、C端点度数、D模体支持；同一划分跨密度查询固定，实际物理密度及约束不足见mask_rates.csv。颜色使用同一对称色标，数字符号提供冗余编码；不据图中最优密度重新选参。

**图4：选择稳定性。** 鲁棒方法有{zero:.1%}的选择为α0，这只能说明未启用调制，不能单凭回退率证明正式任务安全。每图9次为三个既有源端训练种子×三个代理划分，不是9份独立预训练骨干；不同α用颜色与纹理双重编码。

**图5：鲁棒对单环境LCB。** 鲁棒选择相对阶段六原采样LCB在{positives}图上MRR点差为正、{negatives}图上为负，其余为零。点为三个既有训练种子的平均差，主代理种子6101；线为5000次原始事实配对Bootstrap的95%区间，先合并头尾RR，未把两个方向当作独立样本。区间条件于图与检查点，未对图谱来源总体作IID保证；主要图级多重比较结果另见Holm列。

PNG仅用于本报告预览；PDF/SVG为直接生成的矢量导出，不包裹截图。主图按整页宽度设计，若用于论文单栏需重新安排面板，不能直接缩小至字号不可读。
'''
    (dest/'captions.md').write_text(captions,encoding='utf8')
    log('figures_complete',files=sorted(x.name for x in dest.glob('*.png')))


if __name__=='__main__':main()
