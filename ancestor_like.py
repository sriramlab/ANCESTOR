from __future__ import division
import os, sys, random
import numpy as np
import pymc as pm
import scipy.stats
from scipy import stats


##############################################################################
# code for computing the log-likelihood
##############################################################################

def align_blocks(state_a, state_b):
    N = len(state_a)
    prev=[state_a[0], state_b[0]]
    ordered=[[state_a[0],state_b[0]]]
    for i in range(1,N):
        a = state_a[i]
        b = state_b[i]
        if a==prev[0]:
            ordered.append([a,b])
        elif a==prev[1]:
            ordered.append([b,a])
        elif b==prev[0]:
            ordered.append([b,a])
        else: ordered.append([a,b])
        prev=ordered[-1]
    return ordered

def check_params(params):
    lambda_0, lambda_1, alpha_0, alpha_1 = params
    if min(lambda_0) < 0 or min(lambda_1)< 0 or min(alpha_0)<0 or min(alpha_1)<0: return -1
    if lambda_0[0]<lambda_1[0]: return -1
    return 1

def likelihood_new(value, params):
    flag = check_params(params)
    if flag==-1: return -100000
    state_a, state_b, length=value
    lambda_0, lambda_1, alpha_0, alpha_1 = params
    alpha_0 = np.array(alpha_0)
    alpha_1 = np.array(alpha_1)
    
    par_0_exp = lambda_0*(1-alpha_0)
    par_1_exp = lambda_1*(1-alpha_1)
    par_0_coeff = lambda_0*alpha_0
    par_1_coeff = lambda_1*alpha_1
    N = len(state_a)
    exponents = []
    coefficients = []
    ordered = align_blocks(state_a, state_b)
    ordered.append([-1,-1])

    curr_exponents=[[0.0],[0.0]]
    # modify the first blocks
    a, b = ordered[0]
    curr_coefficients=[[1/alpha_0[a], 1/alpha_1[b]],[1/alpha_1[a], 1/alpha_0[b]]]
    for i in range(N):
        a, b = ordered[i]
        L = length[i]
        if a == b:
            exponents.append(curr_exponents)
            coefficients.append(curr_coefficients)
            curr_exponents=[[0.0],[0.0]]
            curr_coefficients=[[1.0],[1.0]]
        curr_exponents[0].append(L*(par_0_exp[a] + par_1_exp[b]))
        curr_exponents[1].append(L*(par_0_exp[b] + par_1_exp[a]))
        if a != ordered[i+1][0]:
            curr_coefficients[0].append(par_0_coeff[a])
            curr_coefficients[1].append(par_1_coeff[a])
        if b != ordered[i+1][1]:
            curr_coefficients[0].append(par_1_coeff[b])
            curr_coefficients[1].append(par_0_coeff[b])
    # modify the end blocks
    curr_coefficients[0]+=[1-alpha_0[a], 1-alpha_1[b]]
    curr_coefficients[1]+=[1-alpha_1[a], 1-alpha_0[b]]
    
    exponents.append(curr_exponents)
    coefficients.append(curr_coefficients)
    
    
    # compute loglike
    log_like=0
    for i in range(len(exponents)):
        exponents[i] = np.array(exponents[i])
        coefficients[i] = np.array(coefficients[i])
        log_like+=np.log(np.exp(-exponents[i][0].sum())*coefficients[i][0].prod()+np.exp(-exponents[i][1].sum())*coefficients[i][1].prod())
    return log_like

def likelihood_agg(value, params):
    M = len(value)
    log_like=0
    for i in range(M):
        log_like += likelihood_new(value[i], params)
    return log_like

def prepare_input(filename):
    file = open(filename,'r').readlines()
    offspring=[]
    current_run = []
    for i in range(len(file)):
        line=file[i].strip().split('\t')
        chr, s1, s2, l = line
        if i == 0:
            current_run = [[int(s1)], [int(s2)], [float(l)]]
            prev = line
            continue
        if chr != prev[0] or (s1 != prev[1] and s1 != prev[2] and s2 != prev[1] and s2 != prev[2]):
            offspring.append(current_run)
            current_run = []
            current_run = [[int(s1)], [int(s2)], [float(l)]]
            prev = line
            continue
        current_run[0].append(int(s1))
        current_run[1].append(int(s2))
        current_run[2].append(float(l))
        prev = line
    offspring.append(current_run)
    return offspring
        
def prepare_input_2pop(filename):
    file = open(filename,'r').readlines()
    offspring=[]
    current_run = []
    for i in range(len(file)):
        line=file[i].strip().split('\t')
        chr, s1, s2, l = line
        if s1 == '2': s1='1'
        if s2 == '2': s2='1'
        if i == 0:
            current_run = [[int(s1)], [int(s2)], [float(l)]]
            prev = line
            continue
        if chr != prev[0] or (s1 != prev[1] and s1 != prev[2] and s2 != prev[1] and s2 != prev[2]):
            offspring.append(current_run)
            current_run = []
            current_run = [[int(s1)], [int(s2)], [float(l)]]
            prev = line
            continue
        current_run[0].append(int(s1))
        current_run[1].append(int(s2))
        current_run[2].append(float(l))
        prev = line
    offspring.append(current_run)
    return offspring


##############################################################################
# code for simulation
##############################################################################

# merge blocks of the same label.
def merge_blocks(P):
    P_merged=[]
    prev_state, prev_pos = P[0]
    for i in range(1,len(P)):
        s,pos = P[i]
        if s!=prev_state: P_merged.append([prev_state, prev_pos])
        prev_state = s
        prev_pos = pos
    P_merged.append([prev_state, prev_pos])
    return P_merged
    
def generate_allele(lambda_0, alpha_0, N):
    P0=[]
    pos0=0
    # generate random samplers for alpha_0 and alpha_1.
    xk = range(len(alpha_0))
    alpha_0_sampler = stats.rv_discrete(name='alpha_0_sampler', values=(xk, alpha_0))
    
    # put down boundaries and labels.
    P0_sizes = np.random.exponential(1, size=N)
    state_0 = alpha_0_sampler.rvs(size=N)
    
    # stretch each block by 1/lambda.
    for i in range(N):
        pos0 += P0_sizes[i]/lambda_0[state_0[i]]
        P0.append([state_0[i], pos0])
    return P0

    
def compute_proportion_for_parameter(lambda_0, alpha_0, N_steps=10000):
    P = generate_allele(lambda_0, alpha_0, N_steps)
    prop = compute_proportion(P, len(lambda_0))
    return prop

def compute_proportion(P, M):
    length = np.array([0.0]*M)
    prev_pos=0
    for i in range(len(P)):
        s, pos = P[i]
        length[s]+=pos-prev_pos
        prev_pos = pos
    return length/np.sum(length)

# assume that the offspring is generated from simulation and is phased.
def compute_proportion_from_offspring(offspring):
    length_0=np.array([0.0]*3)
    length_1=np.array([0.0]*3)
    for i in range(len(offspring[0])):
        s0 = offspring[0][i]
        s1 = offspring[1][i]
        l = offspring[2][i]
        length_0[s0]+=l
        length_1[s1]+=l
    return length_0/np.sum(length_0), length_1/np.sum(length_1)


def simulate_3pop(params):
    lambda_0, lambda_1, alpha_0, alpha_1 = params
    N=400
    P0 = generate_allele(lambda_0, alpha_0, N)
    P1 = generate_allele(lambda_1, alpha_1, N)
    
    # merge blocks with the same label.
    P0_merged = merge_blocks(P0)
    P1_merged = merge_blocks(P1)
    offspring = create_offspring(P0_merged, P1_merged)
    offspring = np.array(offspring)
    state_0 = offspring[:,0]; state_1 = offspring[:,1]; length = offspring[:,2]
    offspring = [state_0, state_1, length]
    return P0_merged, P1_merged, offspring

def compute_posterior_prop(M):
    N = M.trace('lambda_a').length()
    N_sub = 250
    ind = random.sample(range(N), N_sub)
    prop0_trace=[]
    prop1_trace=[]
    for i in ind:   
        lambda_a = M.trace('lambda_a')[i]
        lambda_b = M.trace('lambda_b')[i]
        lambda_c = M.trace('lambda_c')[i]
        lambda_d = M.trace('lambda_d')[i]
        lambda_e = M.trace('lambda_e')[i]
        lambda_f = M.trace('lambda_f')[i]
        alpha_a = M.trace('alpha_a')[i]
        alpha_b = M.trace('alpha_b')[i]
        alpha_c = M.trace('alpha_c')[i]
        alpha_d = M.trace('alpha_d')[i]
        prop0 = compute_proportion_for_parameter([lambda_a, lambda_b, lambda_c], [alpha_a, alpha_b, 1-alpha_a-alpha_b])
        prop1 = compute_proportion_for_parameter([lambda_d, lambda_e, lambda_f], [alpha_c, alpha_d, 1-alpha_c-alpha_d])
        prop0_trace.append(prop0)
        prop1_trace.append(prop1)
    return prop0_trace, prop1_trace
    
def create_offspring(P1, P2):
    i=0
    j=0
    N1 = len(P1)
    N2 = len(P2)
    offspring=[]
    prev_pos=0
    while True:
        s1, pos1 = P1[i]
        s2, pos2 = P2[j]
        if i==N1-1 or j==N2-1: break
        if pos1<=pos2:
            if pos1!=prev_pos: offspring.append([s1, s2, pos1-prev_pos])
            i+=1
            prev_pos = pos1
        else:
            if pos2!=prev_pos:  offspring.append([s1, s2, pos2-prev_pos])
            j+=1
            prev_pos = pos2
    return offspring




##############################################################################
# code for printing
##############################################################################

# print median value for all parameters.
def print_results(M, type='MCMC',write_file=False):
    if type=='MCMC':
        lambda_a = M.lambda_a.stats()['quantiles'][50]
        lambda_b = M.lambda_b.stats()['quantiles'][50]
        lambda_c = M.lambda_c.stats()['quantiles'][50]
        lambda_d = M.lambda_d.stats()['quantiles'][50]
        lambda_e = M.lambda_e.stats()['quantiles'][50]
        lambda_f = M.lambda_f.stats()['quantiles'][50]
        alpha_a = M.alpha_a.stats()['quantiles'][50]
        alpha_b = M.alpha_b.stats()['quantiles'][50]
        alpha_c = M.alpha_c.stats()['quantiles'][50]
        alpha_d = M.alpha_d.stats()['quantiles'][50]
    if type=='MAP':
        lambda_a = M.lambda_a.value
        lambda_b = M.lambda_b.value
        lambda_c = M.lambda_c.value
        lambda_d = M.lambda_d.value
        lambda_e = M.lambda_e.value
        lambda_f = M.lambda_f.value
        alpha_a = M.alpha_a.value
        alpha_b = M.alpha_b.value
        alpha_c = M.alpha_c.value
        alpha_d = M.alpha_d.value
    if write_file == False:
        print 'lambda_a: ', round(lambda_a,2)
        print 'lambda_b: ', round(lambda_b,2)
        print 'lambda_c: ', round(lambda_c,2)
        print 'lambda_d: ', round(lambda_d,2)
        print 'lambda_e: ', round(lambda_e,2)
        print 'lambda_f: ', round(lambda_f,2)
        print 'alpha_a: ', round(alpha_a,2)
        print 'alpha_b: ', round(alpha_b,2)
        print 'alpha_c: ', round(alpha_c,2)
        print 'alpha_d: ', round(alpha_d,2)
        print 'prop1: ', compute_proportion_for_parameter([lambda_a, lambda_b, lambda_c], [alpha_a, alpha_b, 1-alpha_a-alpha_b])
        print 'prop2: ', compute_proportion_for_parameter([lambda_d, lambda_e, lambda_f], [alpha_c, alpha_d, 1-alpha_c-alpha_d])
    if write_file != False:
        out = open(write_file,'a')
        out.write('lambda_a: '+ str(round(lambda_a,2))+'\n')
        out.write('lambda_b: '+ str(round(lambda_b,2))+'\n')
        out.write('lambda_c: '+ str(round(lambda_c,2))+'\n')
        out.write('lambda_d: '+ str(round(lambda_d,2))+'\n')
        out.write('lambda_e: '+ str(round(lambda_e,2))+'\n')
        out.write('lambda_f: '+ str(round(lambda_f,2))+'\n')
        out.write('alpha_a: '+ str(round(alpha_a,2))+'\n')
        out.write('alpha_b: '+ str(round(alpha_b,2))+'\n')
        out.write('alpha_c: '+ str(round(alpha_c,2))+'\n')
        out.write('alpha_d: '+ str(round(alpha_d,2))+'\n')
        prop1 = compute_proportion_for_parameter([lambda_a, lambda_b, lambda_c], [alpha_a, alpha_b, 1-alpha_a-alpha_b])
        out.write('\t'.join([str(e) for e in prop1])+'\n')
        prop2 = compute_proportion_for_parameter([lambda_d, lambda_e, lambda_f], [alpha_c, alpha_d, 1-alpha_c-alpha_d])
        out.write('\t'.join([str(e) for e in prop2])+'\n')
        out.close()
        
        
        
