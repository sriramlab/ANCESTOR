from __future__ import division
import os, sys, random, cPickle
import numpy as np
import pymc as pm
import scipy.stats
from scipy import stats
import ancestor_like

reload(ancestor_like)

def construct_module(filename):
    ancestor_like.offspring = ancestor_like.prepare_input(filename)
    
    ancestor_like.lambda_a = pm.Uniform('lambda_a', lower=0, upper=100)
    ancestor_like.lambda_b = pm.Uniform('lambda_b', lower=0, upper=100)
    ancestor_like.lambda_c = pm.Uniform('lambda_c', lower=0, upper=100)
    ancestor_like.lambda_d = pm.Uniform('lambda_d', lower=0, upper=100)
    ancestor_like.lambda_e = pm.Uniform('lambda_e', lower=0, upper=100)
    ancestor_like.lambda_f = pm.Uniform('lambda_f', lower=0, upper=100)
    
    ancestor_like.alpha_a = pm.Uniform('alpha_a', lower=0, upper=1)
    ancestor_like.alpha_b = pm.Uniform('alpha_b', lower=0, upper=1)
    ancestor_like.alpha_c = pm.Uniform('alpha_c', lower=0, upper=1)
    ancestor_like.alpha_d = pm.Uniform('alpha_d', lower=0, upper=1)
    
    
    ancestor_like.params=[np.array([ancestor_like.lambda_a, ancestor_like.lambda_b, ancestor_like.lambda_c]),np.array([ancestor_like.lambda_d, ancestor_like.lambda_e, ancestor_like.lambda_f]), np.array([ancestor_like.alpha_a, ancestor_like.alpha_b, 1-ancestor_like.alpha_a-ancestor_like.alpha_b]), np.array([ancestor_like.alpha_c, ancestor_like.alpha_d, 1-ancestor_like.alpha_c-ancestor_like.alpha_d])]
    
    
    ancestor_like.data = pm.Stochastic(logp = ancestor_like.likelihood_agg, doc='observed data', name = 'data', parents = {'params':ancestor_like.params}, value=ancestor_like.offspring, dtype=list, observed=True, plot=True)
    return ancestor_like

write_file_name = 'summary_inference.txt'

def run_mcmc(filename, outname, iter = 9000, burn = 1000, progress_bar=False, thin = 5):
    print filename
    #out = open(write_file_name,'a')
    #out.write(filename+'\n'); out.close()
    test = construct_module(filename)
    M_map = pm.MAP(test)
    M_map.fit()
    M_mcmc = pm.MCMC(test)
    M_mcmc.sample(iter=iter, burn=burn, progress_bar=progress_bar, thin=thin)
    
    M_mcmc.lambda_a.value = M_mcmc.lambda_a.stats()['quantiles'][50]
    M_mcmc.lambda_b.value = M_mcmc.lambda_b.stats()['quantiles'][50]
    M_mcmc.lambda_c.value = M_mcmc.lambda_c.stats()['quantiles'][50]
    M_mcmc.lambda_d.value = M_mcmc.lambda_d.stats()['quantiles'][50]
    M_mcmc.lambda_e.value = M_mcmc.lambda_e.stats()['quantiles'][50]
    M_mcmc.lambda_f.value = M_mcmc.lambda_f.stats()['quantiles'][50]
    M_mcmc.alpha_a.value = M_mcmc.alpha_a.stats()['quantiles'][50]
    M_mcmc.alpha_b.value = M_mcmc.alpha_b.stats()['quantiles'][50]
    M_mcmc.alpha_c.value = M_mcmc.alpha_c.stats()['quantiles'][50]
    M_mcmc.alpha_d.value = M_mcmc.alpha_d.stats()['quantiles'][50]
    
    M_map = pm.MAP(test)
    M_map.fit()
    
    #ancestor_like.print_results(M_mcmc, write_file = write_file_name)
    results={}
    results['lambda_a'] = M_mcmc.lambda_a.stats()
    results['lambda_b'] = M_mcmc.lambda_b.stats()
    results['lambda_c'] = M_mcmc.lambda_c.stats()
    results['lambda_d'] = M_mcmc.lambda_d.stats()
    results['lambda_e'] = M_mcmc.lambda_e.stats()
    results['lambda_f'] = M_mcmc.lambda_f.stats()
    results['alpha_a'] = M_mcmc.alpha_a.stats()
    results['alpha_b'] = M_mcmc.alpha_b.stats()
    results['alpha_c'] = M_mcmc.alpha_c.stats()
    results['alpha_d'] = M_mcmc.alpha_d.stats()
    results['MAP']=[M_map.lambda_a.value, M_map.lambda_b.value, M_map.lambda_c.value,
                    M_map.lambda_d.value, M_map.lambda_e.value, M_map.lambda_f.value,
                    M_map.alpha_a.value, M_map.alpha_b.value, M_map.alpha_c.value, M_map.alpha_d.value]
    results['trace'] = [M_mcmc.lambda_a.trace(), M_mcmc.lambda_b.trace(), M_mcmc.lambda_c.trace(),
                        M_mcmc.lambda_d.trace(), M_mcmc.lambda_e.trace(), M_mcmc.lambda_f.trace(),
                        M_mcmc.alpha_a.trace(), M_mcmc.alpha_b.trace(), M_mcmc.alpha_c.trace(), M_mcmc.alpha_d.trace()]
    
    #return M_map, M_mcmc
    cPickle.dump(results, open(outname,'wb'))
    
def extract_parent_proportions_new(inf_dict_name, analysis='median'):
    inf_dict = cPickle.load(open(inf_dict_name,'rb'))
    if analysis == 'median':
        lambda_a = inf_dict['lambda_a']['quantiles'][50]
        lambda_b = inf_dict['lambda_b']['quantiles'][50]
        lambda_c = inf_dict['lambda_c']['quantiles'][50]
        lambda_d = inf_dict['lambda_d']['quantiles'][50]
        lambda_e = inf_dict['lambda_e']['quantiles'][50]
        lambda_f = inf_dict['lambda_f']['quantiles'][50]
        alpha_a = inf_dict['alpha_a']['quantiles'][50]
        alpha_b = inf_dict['alpha_b']['quantiles'][50]
        alpha_c = inf_dict['alpha_c']['quantiles'][50]
        alpha_d = inf_dict['alpha_d']['quantiles'][50]
    propA = ancestor_like.compute_proportion_for_parameter([lambda_a, lambda_b, lambda_c], [alpha_a, alpha_b, 1-alpha_a-alpha_b], N_steps=10000)
    propB = ancestor_like.compute_proportion_for_parameter([lambda_d, lambda_e, lambda_f], [alpha_c, alpha_d, 1-alpha_c-alpha_d], N_steps=10000)
    return propA, propB

def run_ancestor(filename, outname, iter = 5000, burn = 1000, progress_bar=False, thin = 5):
    run_mcmc(filename, outname, iter=iter, burn=burn, progress_bar=progress_bar, thin=thin)
    propA, propB = extract_parent_proportions_new(outname, analysis='median')
    print ''
    print "Parent 1's genomic ancestry:", np.round(propA.tolist(),3)
    print "Parent 2's genomic ancestry:", np.round(propB.tolist(),3)
    
if __name__ == '__main__':
    run_ancestor(sys.argv[1], sys.argv[2])