"""Condensed two-page research note. Full detail lives in research_note.md."""
from pathlib import Path
import json
import pandas as pd
from reportlab import rl_config
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.pagesizes import A4
from .common import ROOT,RESULTS

INK=colors.HexColor('#18343C');TEAL=colors.HexColor('#007F7A');MUTED=colors.HexColor('#526A71');PALE=colors.HexColor('#EDF5F4')

def main():
    rl_config.invariant=1
    fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('AuditSans','DejaVuSans.ttf'),('AuditSans-Bold','DejaVuSans-Bold.ttf'),('AuditSans-Oblique','DejaVuSans-Oblique.ttf'),('AuditSans-BoldOblique','DejaVuSans-BoldOblique.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
    pdfmetrics.registerFontFamily('AuditSans',normal='AuditSans',bold='AuditSans-Bold',italic='AuditSans-Oblique',boldItalic='AuditSans-BoldOblique')
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    path=out/'Trusting_Simulation_Research_Note.pdf'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Eyebrow2',fontName='AuditSans-Bold',fontSize=8,leading=10,textColor=TEAL,spaceAfter=10))
    styles.add(ParagraphStyle(name='Title2',fontName='AuditSans-Bold',fontSize=24,leading=28,textColor=INK,spaceAfter=12))
    styles.add(ParagraphStyle(name='Lead2',fontName='AuditSans',fontSize=11.5,leading=15.7,textColor=INK,spaceAfter=13))
    styles.add(ParagraphStyle(name='Body2',fontName='AuditSans',fontSize=9.4,leading=13,textColor=INK,spaceAfter=7))
    styles.add(ParagraphStyle(name='Small2',fontName='AuditSans',fontSize=7.7,leading=10.4,textColor=MUTED,spaceAfter=6))
    styles.add(ParagraphStyle(name='Head2',fontName='AuditSans-Bold',fontSize=11,leading=14,textColor=INK,spaceBefore=8,spaceAfter=6))
    styles.add(ParagraphStyle(name='Table2',fontName='AuditSans',fontSize=8.3,leading=11,textColor=INK))
    def p(t,style='Body2'):return Paragraph(t,styles[style])
    story=[]
    story += [p('INDEPENDENT PUBLIC-DATA AUDIT  /  24 OCTOBER 2026','Eyebrow2'),
      p('When can simulation predict<br/>real-world robot rankings?','Title2'),
      p('Matched simulation is useful on SIMPLER. This audit finds no improvement from the prespecified task-aware calibrator, and no validated rule for skipping physical tests.','Lead2')]
    summary=Table([[p('<b>42</b> paired score cells','Body2'),p('<b>9</b> task entries','Body2'),p('<b>2</b> robot embodiments','Body2')]],colWidths=[166,166,166])
    summary.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),10)]))
    story += [summary,Spacer(1,7),p('Held-out evaluation','Head2'),
      p('Every fitted prediction excludes all physical labels from its test task and stays within robot embodiment. B1 uses matched simulation; B2 is affine calibration; C1 adds two frozen task attributes with fixed shrinkage. No hyperparameters are chosen on test outcomes.')]
    summarydf=pd.read_csv(RESULTS/'simpler_summary.csv').set_index(['embodiment','method'])
    table=[[p('<font color="#FFFFFF"><b>Method</b></font>','Table2'),p('<font color="#FFFFFF">Google<br/>rank error</font>','Table2'),p('<font color="#FFFFFF">Google<br/>MAE</font>','Table2'),p('<font color="#FFFFFF">WidowX<br/>rank error</font>','Table2'),p('<font color="#FFFFFF">WidowX<br/>MAE</font>','Table2')]]
    names={'B0':'Global simulation','B1':'Matched simulation','B2':'Affine calibration','C1':'Task attributes'}
    for method in ['B0','B1','B2','C1']:
        g=summarydf.loc['google_robot',method];w=summarydf.loc['widowx',method]
        table.append([p(f'{method}  {names[method]}','Table2'),p(f'{g.ranking_error*100:.1f}%','Table2'),p(f'{g.mae*100:.1f} pp','Table2'),p(f'{w.ranking_error*100:.1f}%','Table2'),p(f'{w.mae*100:.1f} pp','Table2')])
    tab=Table(table,colWidths=[174,81,81,81,81],hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('TEXTCOLOR',(0,0),(-1,0),colors.white),
      ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F4F7F8'),colors.white]),
      ('BACKGROUND',(0,2),(-1,2),PALE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
      ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    # Paragraph text color must be set explicitly for the dark header.
    for col in table[0]:col.style=ParagraphStyle('WhiteHeader',parent=styles['Table2'],textColor=colors.white,fontName='AuditSans-Bold')
    story += [tab,Spacer(1,5),p('Rank error averages tasks within embodiment; only observed real gaps of at least 10 points are eligible. MAE is mean absolute error in percentage points. Five Google and four WidowX tasks underpin these estimates.','Small2'),
      p('What the failures reveal','Head2'),
      p('<b>B1 gets 61/65 Google and 8/9 WidowX comparisons right.</b> These are overlapping pairs, not independent samples. Closing drawers accounts for three material ranking errors. Move Near and Carrot on Plate each add one.'),
      p('On Move Near, simulation slightly favors RT-1-Begin over Octo-Base (5.0% vs 4.2%); physical success instead favors Octo-Base by 33.3 points. A strong overall correlation can hide consequential local mistakes.'),
      p('<b>C1 cannot repair individual rankings when its simulation slope is positive.</b> All fitted slopes are positive. Task-only additive effects therefore test calibration, not richer policy-task interactions. C1 also fails to beat B1 on MAE; this is a null for this model, not a general verdict on task-aware methods.'),
      p('All nine Pearson/MMRV outputs match the pinned authors\' functions. Paper/code differences are documented: the paper includes an extra Google policy and two different eggplant outcomes. The repository snapshot remains primary. Unknown trial counts are left blank. LOTO holds out calibrator labels, not the original simulator-design process.','Small2'),PageBreak()]
    story += [p('EXTERNAL AUDIT  /  PHYSICAL STRESS TEST','Eyebrow2'),
      p('Task mix can change the winner','Title2')]
    a=json.loads((RESULTS/'robocurve_aggregate_difference.json').read_text())
    story += [p(f'Robocurve\'s 360 trials put Astra at 36.69% progress and Opus 5.5 at 36.01%. Their difference is {a["astra_minus_opus55"]*100:+.2f} points (conditional 95% bootstrap interval: {a["ci_low"]*100:+.2f} to {a["ci_high"]*100:+.2f}).','Body2')]
    fig=RESULTS/'figures/04_robocurve_physical_stress_test.png'
    from PIL import Image as PILImage
    with PILImage.open(fig) as im:w,h=im.size
    story += [Image(str(fig),width=498,height=498*h/w),Spacer(1,7),
      p('Astra leads Stack Bowls by 22.5 progress points; Opus 5.5 leads Cap Pen by 16. Omitting Stack Bowls or Store In Safe flips the overall leader. Full completions are only 5/120 for Astra and 1/120 for Opus 5.5. Partial progress and completion remain separate.','Body2'),
      p('RoboDojo: scores found, compatibility still fails','Head2'),
      p('<b>Correction:</b> paper Table 2 provides 180 physical method-task cells (10 methods, 18 tasks; 10 trials per cell). The first website-only audit missed these. But the paper explicitly uses unpaired tasks and separate domain fine-tuning. Shared method names do not identify the same evaluated checkpoint. No matched-task sim-to-real validation is claimed.'),
      p('As a descriptive cross-task method proxy, overall simulation rankings have 26.8% task-average completion-ranking error on Piper. Piper X has 0% error across only two informative tasks. These are observed gaps, not confident true orderings; one success changes the rate by 10 points.','Small2'),
      p('The next resource is compatible evaluation evidence','Head2'),
      p('Request existing same-checkpoint results on explicitly paired tasks, plus rubric/controller/layout versions. Inspect Cover Blocks and Insert Tubes on ARX X5 first. Request raw repeated-round scores for Store in Safe and Stack Bowls; aggregate stability can conceal task variation.'),
      p('If a new study becomes necessary, a 12-task, three-family, three-policy pilot with 20 physical trials per cell would total 720 trials. That is a feasibility pilot, not sufficient precision for most 10-point gaps. Freeze a representative task mix and evaluate a trust rule on held-out tasks against B1.'),
      p('Limits and reproduction','Head2'),
      p('Tasks and checkpoints are few and correlated. Robocurve selected tasks using Astra and uses non-blinded grading and manual resets. Its hardware, agent scaffold and rubrics differ from original RoboDojo. Bootstrap intervals condition on these six tasks and omit rig/time dependence. The exploratory trust score did not yield a defensible deferral rule.','Small2'),
      p('Run <b>python -m src.reproduce</b> from the source bundle. Full tables, five figures, source hashes, leakage tests and recommendations are included. No new robot trials, simulation, policy inference or paid APIs were used. Not commissioned or endorsed by Robocurve.','Small2'),
      p('Sources: <link href="https://proceedings.mlr.press/v270/li25c.html" color="#007F7A">SIMPLER paper</link> | <link href="https://github.com/simpler-env/SimplerEnv/blob/eddb5690749a1e34267807c0f7b5eede323918d7/simpler_env/utils/metrics.py" color="#007F7A">pinned constants and metrics</link> | <link href="https://arxiv.org/html/2607.04434v3" color="#007F7A">RoboDojo paper (Tables 1, 2, 5, 9)</link> | <link href="https://robocurve.org/opus-5-5-robodojo-rc-tier-1/" color="#007F7A">Robocurve report and trial data</link>','Small2')]
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CBD9DA'));canvas.line(48,39,547,39)
        canvas.setFillColor(MUTED);canvas.setFont('AuditSans',7.3)
        canvas.drawString(48,27,'TRUSTING SIMULATION  |  Independent research audit  |  2026-10-24')
        canvas.drawRightString(547,27,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=49,
        title='When can simulation predict real-world robot rankings?',author='Independent research audit')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)
if __name__=='__main__':main()
