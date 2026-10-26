"""Task-balanced calibrators; all fitting receives only training-task labels."""
import numpy as np
from .taxonomy import FEATURES

METHODS=['B0','B0_exclude','B1','B2','B2_ridge','C1']

def design(frame,features):
    return np.column_stack([np.ones(len(frame)),frame.sim.to_numpy(),*[frame[f].to_numpy() for f in features]])

def weights(frame):
    counts=frame.groupby('task_id').task_id.transform('size').to_numpy()
    w=1/counts
    return w/w.sum()

def fit_calibrator(train,features=(),penalty=0):
    x=design(train,features);w=weights(train);y=train.real.to_numpy()
    p=np.diag([0]+[penalty]*(x.shape[1]-1))
    return np.linalg.pinv(x.T@(w[:,None]*x)+p)@(x.T@(w*y))

def predict_fold(train,test,all_sim,method):
    # all_sim must not contain real labels. Only simulation and identity matter.
    if 'real' in all_sim:raise ValueError('all_sim must be label-free')
    if set(train.task_id)&set(test.task_id):raise ValueError('Test task labels entered fitting')
    if method=='B1':return test.sim.to_numpy(),None
    if method.startswith('B0'):
        use=all_sim
        if method=='B0_exclude':use=use[~use.task_id.isin(test.task_id)]
        averages=use.groupby('policy_id').sim.mean()
        return test.policy_id.map(averages).to_numpy(),None
    features=FEATURES if method=='C1' else []
    penalty=0.10 if method in {'C1','B2_ridge'} else 0.0
    beta=fit_calibrator(train,features,penalty)
    return design(test,features)@beta,beta
