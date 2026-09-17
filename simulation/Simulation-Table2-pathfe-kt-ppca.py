"""Table 2: PathFE, KnowledgeTransfer and PerPCA on Simulation-3."""
import argparse, os, random, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
HERE=os.path.dirname(os.path.abspath(__file__)); ALG=os.path.join(os.path.dirname(HERE), 'src', 'distributedpca')
sys.path.insert(0, ALG)
from feature_extraction_path import PathFE
from knowledge_transfer import KnowledgeTransfer
from pca_utils import PCAUtils

def proj(f,p):
    f=np.asarray(f,float)
    if f.size==0:return np.zeros((p,p))
    if f.ndim==1:f=f[:,None]
    if f.shape[0]!=p:f=f.T
    return f@np.linalg.pinv(f.T@f)@f.T

def one(seed,p,args):
    n=p; np.random.seed(seed); random.seed(seed)
    covs=PCAUtils.generate_cov_matrix(args.n_clients, args.orthonormal, [8,8,4], args.lam, p, 2)
    ns = [max(2, int(round(n*m))) for m in np.linspace(.6, 1.4, args.n_clients)] if args.unequal_samples else [n]*args.n_clients
    X=[np.random.multivariate_normal(np.zeros(p), cov, size=ni) for cov,ni in zip(covs,ns)]
    true=np.sqrt(p)*np.linalg.svd(covs[0])[0][:,:2]; pt=proj(true,p)
    # True local direction is the leading eigenvector in the complement.
    locals=[]; comp=np.eye(p)-pt
    for cov in covs:
        val,vec=np.linalg.eigh(comp@cov@comp); locals.append(vec[:,np.argsort(val)[::-1][:1]]*np.sqrt(p))
    eps=n**(0.5-2/args.a-args.epsilon0); ranks=[3]*args.n_clients
    path=PathFE().fit(X,eps,num_list=ranks,epsilon0=args.epsilon0,a=args.a,pair_n_method=args.pair_n_method)
    kt=KnowledgeTransfer(ranks,target_index=0,shared_rank=2,target_rank=2).fit([z.copy() for z in X])
    if not args.perpca_path: raise RuntimeError('PERPCA_PATH is required')
    if args.perpca_path not in sys.path:sys.path.insert(0,args.perpca_path)
    from algs import personalized_pca_dgd
    U,V,_=personalized_pca_dgd(X,{'ngc':2,'nlc':1,'num_client':args.n_clients,'global_epochs':args.ppca_epochs,'eta':.1,'rho':1.})
    estimates={'pathfe':(path.global_factors,path.est_pf,path.num_cf),'kt':(kt.global_factors,kt.local_factor_list,kt.num_common),'ppca':(U[0]*np.sqrt(p),V,2)}
    out=[]
    for method,(gf,lf,nc) in estimates.items():
        pg=proj(gf,p); tr=float(np.trace(pg@pt)/2); common_norm=sum(np.linalg.norm(x@pt,'fro')**2 for x in X)
        diff=sum(np.linalg.norm(x@pg-x@pt,'fro')**2 for x in X)
        row={'method':method,'num_common':nc,'tr':tr,'scaled_projection_distance':np.sqrt(max(0,1-tr)),
             'global_subspace_error':np.linalg.norm(pg-pt,'fro')**2,
             'common_component_rrmse':np.sqrt(diff/max(common_norm,1e-12))}
        if lf is None: row.update(local_subspace_error=np.nan,subspace_error=np.nan,reconstruction_error=np.nan)
        else:
            le=[]; re=[]
            for x,tl,el in zip(X,locals,lf):
                pl=proj(el,p); ptl=proj(tl,p); le.append(np.linalg.norm(pl-ptl,'fro')**2)
                re.append(np.linalg.norm(x-x@(pg+pl),'fro')**2/x.shape[0])
            row['local_subspace_error']=float(np.mean(le)); row['subspace_error']=row['global_subspace_error']+row['local_subspace_error']; row['reconstruction_error']=float(np.mean(re))
        out.append(row)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=12345); ap.add_argument('--n-runs',type=int,default=50); ap.add_argument('--n-jobs',type=int,default=8); ap.add_argument('--p-list',type=int,nargs='+',default=[100,300,500]); ap.add_argument('--n-clients',type=int,default=10); ap.add_argument('--unequal-samples',action='store_true'); ap.add_argument('--lam',type=float,nargs=3,default=[15,8,3]); ap.add_argument('--epsilon0',type=float,default=.1); ap.add_argument('--a',type=float,default=10.); ap.add_argument('--pair-n-method',choices=('min','harmonic'),default='min'); ap.add_argument('--ppca-epochs',type=int,default=50); ap.add_argument('--perpca-path',default=os.environ.get('PERPCA_PATH', os.path.join(os.path.dirname(HERE), 'local', 'perpca'))); ap.add_argument('--orthonormal',action=argparse.BooleanOptionalAction,default=True); ap.add_argument('--output',default='results/simulation/Table2-pathfe-kt-ppca.csv'); args=ap.parse_args()
    rows=[]
    for p in args.p_list:
        vals=Parallel(n_jobs=args.n_jobs)(delayed(one)(args.seed+i,p,args) for i in range(args.n_runs))
        for method in ('pathfe','kt','ppca'):
            d=[x for v in vals for x in v if x['method']==method]; row={'p':p,'method':method,'n_runs':args.n_runs}
            for k in d[0]:
                if k not in ('method','num_common'): row[k]=float(np.nanmean([x[k] for x in d]))
            row['estimated_common_mean']=float(np.mean([x['num_common'] for x in d])); row['estimated_common_std']=float(np.std([x['num_common'] for x in d])); rows.append(row)
    df=pd.DataFrame(rows); os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True); df.to_csv(args.output,index=False); print(df.to_string(index=False))
if __name__=='__main__':main()
