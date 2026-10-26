"""Frozen name-based taxonomy, written before inspecting SIMPLER outcomes.

Primary fitted features: articulation and precision_contact only.
All labels are manually specified working judgments, not benchmark annotations.
"""
from pathlib import Path
import pandas as pd

FEATURES = ['articulation', 'precision_contact']

def attributes(task):
    name = task.lower()
    drawer = 'drawer' in name
    grasp = any(k in name for k in ['pick', 'spoon', 'carrot', 'stack', 'eggplant', 'apple'])
    placement = any(k in name for k in ['move_near', 'spoon', 'carrot', 'stack', 'eggplant', 'apple'])
    precision = drawer or any(k in name for k in ['stack', 'carrot'])
    multistep = 'apple' in name
    return dict(grasp=int(grasp), placement=int(placement), articulation=int(drawer),
                precision_contact=int(precision), multistep=int(multistep), deformable=0,
                task_family=('articulation' if drawer else 'placement' if placement else 'grasp'),
                taxonomy_confidence='tentative',
                description='Frozen task-name rubric; drawer/plate precision varied in sensitivity')

def metadata(tasks):
    rows=[]
    for task in sorted(tasks):
        rows.append(dict(source='simpler',task_id=task,
                         embodiment='google_robot' if task.startswith('google') else 'widowx',
                         official_category='not encoded in constants', **attributes(task)))
    return pd.DataFrame(rows)
