"""Standalone exportable research figures; all inputs are offline CSV results."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from .common import RESULTS

FIG=RESULTS/'figures'
COLORS={'B0':'#809096','B1':'#007F7A','B2':'#D48C24','C1':'#7961A8'}
LABELS={'B1':'B1  Matched simulation','B2':'B2  Global calibration','C1':'C1  Task-aware calibration'}
NAMES={
 'google_robot_close_drawer':'Close drawer', 'google_robot_move_near':'Move near',
 'google_robot_open_drawer':'Open drawer','google_robot_pick_coke_can':'Pick Coke can',
 'google_robot_place_apple_in_closed_top_drawer':'Apple into closed drawer',
 'widowx_carrot_on_plate':'Carrot on plate','widowx_put_eggplant_in_basket':'Eggplant in basket',
 'widowx_spoon_on_towel':'Spoon on towel','widowx_stack_cube':'Stack cube'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.labelcolor':'#294047','text.color':'#182F36','axes.titleweight':'bold','axes.edgecolor':'#BEC8CA',
 'xtick.color':'#496168','ytick.color':'#496168','savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def predictions():
    pred=pd.read_csv(RESULTS/'simpler_predictions.csv');tasks=sorted(pred.task_id.unique())
    fig,axs=plt.subplots(3,3,figsize=(12,10),sharex=True,sharey=True)
    for ax,task in zip(axs.flat,tasks):
        g=pred[pred.task_id==task];emb='Google' if task.startswith('google') else 'WidowX'
        for method,marker in [('B1','o'),('B2','^'),('C1','s')]:
            d=g[g.method==method]
            ax.scatter(np.clip(d.prediction,0,1),d.real,c=COLORS[method],s=45,marker=marker,alpha=.8,label=LABELS[method],linewidths=.3,edgecolors='white')
        ax.plot([0,1],[0,1],color='#BBC7C8',ls='--',lw=1,zorder=0)
        ax.set_title(f'{NAMES[task]}',fontsize=11,pad=15)
        ax.text(.0,1.02,f'{emb}  |  n = {g.policy_id.nunique()} policy scores',transform=ax.transAxes,fontsize=8,color='#6E7D81')
        ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xticks=[0,.5,1],yticks=[0,.5,1]);ax.grid(alpha=.12)
    for ax in axs[-1]:ax.set_xlabel('Predicted physical success')
    for ax in axs[:,0]:ax.set_ylabel('Observed physical success')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle('SIMPLER: held-out task calibration',x=.065,ha='left',fontsize=19,y=1.015)
    fig.text(.065,.976,'B2/C1 fit only other tasks within the same embodiment. B1 uses the matched simulation score.',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.96));save(fig,'01_simpler_predictions')

def task_metrics():
    met=pd.read_csv(RESULTS/'simpler_metrics.csv');tasks=list(NAMES)
    fig,axs=plt.subplots(1,2,figsize=(13,6.2),gridspec_kw={'width_ratios':[1,1.1]})
    y=np.arange(len(tasks))
    for ax,metric,title in zip(axs,['ranking_error','mae'],['Ranking error on material gaps','Error in predicted success rate']):
        for j,method in enumerate(['B1','B2','C1']):
            d=met[met.method==method].set_index('task_id').loc[tasks]
            ax.barh(y+(j-1)*.23,100*d[metric],height=.21,color=COLORS[method],label=LABELS[method])
        ax.set_yticks(y);ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=12)
        ax.xaxis.grid(alpha=.15);ax.set_axisbelow(True)
        ax.axhline(4.5,color='#9DAEB3',lw=.8,ls='--')
    b1=met[met.method=='B1'].set_index('task_id').loc[tasks]
    axs[0].set_yticklabels([f'{NAMES[t]}  ({int(b1.loc[t,"n_eligible"])} pairs)' for t in tasks],fontsize=9)
    axs[1].set_yticklabels([]);axs[0].set_xlabel('Within-task ranking error (%)');axs[1].set_xlabel('Mean absolute error (percentage points)')
    axs[0].set_xlim(0,57);axs[1].set_xlim(0,36)
    fig.suptitle('Task attributes did not improve the prespecified baseline',x=.03,ha='left',fontsize=18,y=1.02)
    fig.text(.03,.96,'Material gap: observed physical difference >= 10 points. Positive affine calibrations preserve the B1 ranking.',fontsize=10)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.015))
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'02_task_ranking_and_calibration')

def coverage():
    d=pd.read_csv(RESULTS/'robodojo_coverage_matrix.csv')
    tasks=d.real_task.drop_duplicates().tolist();models=sorted(d.policy_id.unique())
    d['level']=np.select([d.same_task_name & d.same_embodiment,d.same_task_name,d.sim_task_score_available],[3,2,1],default=0)
    table=d.pivot(index='policy_id',columns='real_task',values='level').loc[models,tasks]
    palette=['#E7ECEE','#9CA9BF','#D9AB55','#4B9695']
    fig,(avail,ax)=plt.subplots(1,2,figsize=(14,8),sharey=True,gridspec_kw={'width_ratios':[.65,12],'wspace':.05})
    availability=d.groupby('policy_id').real_task_score_available.all().reindex(models).astype(int).to_numpy()[:,None]
    avail.imshow(availability,cmap=ListedColormap(['#E7ECEE','#007F7A']),vmin=0,vmax=1,aspect='auto')
    avail.set_xticks([0]);avail.set_xticklabels(['July real\nscores'],fontsize=9)
    avail.set_yticks(range(len(models)));avail.set_yticklabels(models,fontsize=9)
    for i,value in enumerate(availability[:,0]):avail.text(0,i,'18' if value else '0',ha='center',va='center',fontsize=8,color='white' if value else '#496168')
    ax.imshow(table.to_numpy(),cmap=ListedColormap(palette),vmin=0,vmax=3,aspect='auto')
    ax.set_xticks(range(len(tasks)));ax.set_xticklabels([x.replace('_',' ').title() for x in tasks],rotation=52,ha='right',fontsize=9)
    ax.tick_params(axis='y',labelleft=False,left=False)
    ax.set_xticks(np.arange(-.5,len(tasks),1),minor=True);ax.set_yticks(np.arange(-.5,len(models),1),minor=True)
    ax.grid(which='minor',color='white',lw=1.2);ax.tick_params(which='minor',bottom=False,left=False)
    labels=['No task proxy','Category proxy only','Same name; different hardware','Same name + hardware; protocols unpaired']
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(palette,labels)],loc='lower center',bbox_to_anchor=(.55,.005),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('RoboDojo: physical scores exist, but 0 exact sim-real pairs',x=.03,ha='left',fontsize=19,y=1.02)
    fig.text(.03,.965,'Paper: 180 physical cells across 10 methods. Website: 3,320 simulation cells and 12 shared method names.',fontsize=10)
    fig.text(.03,.925,'Left: July physical-score availability. Matrix: October task proxies. Tasks are unpaired; methods are trained by domain.',fontsize=10,color='#85602E')
    fig.subplots_adjust(left=.17,right=.99,bottom=.40,top=.86);save(fig,'03_robodojo_coverage')

def robodojo_proxy():
    d=pd.read_csv(RESULTS/'robodojo_paper_proxy_task_metrics.csv')
    d=d[d.category=='Average'];tasks=d[d.metric=='success_rate'].task_id.tolist()
    fig,axs=plt.subplots(1,2,figsize=(12.8,8.4),sharey=True)
    for ax,metric,title in zip(axs,['progress','success_rate'],['Partial progress','Task completion']):
        x=d[d.metric==metric].set_index('task_id').loc[tasks]
        for i,r in enumerate(x.itertuples()):
            if r.n_material:
                ax.barh(i,100*r.ranking_error,color='#007F7A',height=.65)
                ax.text(max(100*r.ranking_error,0)+1.5,i,f'{r.n_wrong}/{r.n_material}',va='center',fontsize=8)
            else:
                ax.text(1.5,i,'No eligible pairs',va='center',fontsize=8,color='#7C898D')
        ax.set_xlim(0,82);ax.set_ylim(len(tasks)-.5,-.5)
        ax.set_title(title,loc='left',fontsize=12);ax.set_xlabel('Task ranking error (%)')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for v in [5.5,11.5]:ax.axhline(v,color='#9DAEB3',lw=.8,ls='--')
    axs[0].set_yticks(range(len(tasks)));axs[0].set_yticklabels([t.replace('_',' ').title() for t in tasks],fontsize=9)
    fig.suptitle('Overall simulation ranking is an imperfect physical-task proxy',x=.02,ha='left',fontsize=18,y=.99)
    fig.text(.02,.945,'10 shared methods; separate domain training and unpaired tasks. This is not matched sim-to-real validation.',fontsize=10)
    fig.text(.02,.91,'Bars: error among observed physical gaps >= 10 points. Labels: wrong / eligible pairs; 10 trials per method-task.',fontsize=9)
    fig.text(.02,.02,'ARX X5 / Piper / Piper X task groups, top to bottom. July paper tables only. No-error tasks can still have very limited coverage.',fontsize=9,color='#85602E')
    fig.tight_layout(rect=(0,.05,1,.89));save(fig,'05_robodojo_method_proxy')

def robocurve():
    diff=pd.read_csv(RESULTS/'robocurve_differences.csv');mix=pd.read_csv(RESULTS/'robocurve_task_mix.csv')
    fig,axs=plt.subplots(1,2,figsize=(12.8,4.8),gridspec_kw={'width_ratios':[1.1,1]})
    y=np.arange(len(diff))
    colors=['#007F7A' if x>0 else '#7961A8' for x in diff.astra_minus_opus55]
    for i,r in enumerate(diff.itertuples()):
        axs[0].errorbar(100*r.astra_minus_opus55,i,xerr=[[100*(r.astra_minus_opus55-r.ci_low)],[100*(r.ci_high-r.astra_minus_opus55)]],fmt='o',color=colors[i],capsize=3,ms=7)
    axs[0].set_yticks(y);axs[0].set_yticklabels(diff.task_id.replace({'Classify Objects':'Sort Objects'}),fontsize=9);axs[0].invert_yaxis()
    axs[0].set_title('Task-specific progress differences',loc='left',fontsize=12)
    vals=mix.astra_minus_opus55*100
    axs[1].barh(np.arange(len(mix)),vals,color=['#293E46' if t=='none' else '#007F7A' if v>=0 else '#7961A8' for t,v in zip(mix.omitted_task,vals)],height=.6)
    axs[1].set_yticks(np.arange(len(mix)));axs[1].set_yticklabels(['All six tasks' if t=='none' else 'Omit '+t.replace('Classify Objects','Sort Objects') for t in mix.omitted_task],fontsize=9);axs[1].invert_yaxis()
    axs[1].set_title('Changing the task mix flips the leader',loc='left',fontsize=12)
    for ax in axs:
        ax.axvline(0,c='#657A80',ls='--',lw=1);ax.xaxis.grid(alpha=.15);ax.set_axisbelow(True)
        ax.set_xlabel('Astra minus Opus 5.5 (progress points)')
    axs[0].set_xlim(-30,45);axs[1].set_xlim(-5,5)
    fig.suptitle('Robocurve: physical-only stress test',x=.02,ha='left',fontsize=19,y=1.04)
    fig.text(.02,.965,'20 trials per model-task. Bars on the left are 95% unpaired bootstrap intervals conditional on these six selected tasks.',fontsize=9)
    fig.text(.02,-.015,'Progress is partial-credit performance, not completion. This comparison does not validate simulation.',fontsize=10,color='#85602E')
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'04_robocurve_physical_stress_test')

def main():
    FIG.mkdir(exist_ok=True,parents=True);predictions();task_metrics();coverage();robocurve();robodojo_proxy()
    print('Five figures written (three central + two exploratory supplements), PNG and SVG.')
if __name__=='__main__':main()
