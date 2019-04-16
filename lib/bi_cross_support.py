import numpy as np
import time, sys
from IPython.display import clear_output

def update_progress(progress):
	bar_length = 20
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
	if progress < 0:
		progress = 0
	if progress >= 1:
		progress = 1

	block = int(round(bar_length * progress))
	clear_output(wait = True)
	text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
	print(text)

def break_X_into_ABCD(X,d_n_columns,d_n_rows,):
    
    #returns A,B,C,D
    return X[d_n_columns:,d_n_rows:],X[:d_n_columns,d_n_rows:],X[d_n_columns:,:d_n_rows],X[:d_n_columns,:d_n_rows]

def predict(y_stack,**kwargs):
    options = {
        'k' : 2,
        'd_n_rows'     : int(y_stack.shape[0]/2),
        'd_n_cols'     : int(y_stack.shape[1]/2),
        'SVD_package'  : 'numpy'}

    options.update(kwargs)

    A,B,C,D = break_X_into_ABCD(y_stack,options['d_n_rows'],options['d_n_cols'])
    
    if(options['SVD_package'] == 'sklearn'):

        my_svd = TruncatedSVD(n_components=options['k'], n_iter=7, random_state=42)
        my_svd.fit(D) 
        reconstructed_D = np.dot(my_svd.fit_transform(D),my_svd.components_)
        
    else:
        Ud,Sd,Vd = np.linalg.svd(D)
        Sd[options['k']:] = 0
        reconstructed_D = np.dot(np.dot(Ud,np.diag(Sd)),Vd[:len(Sd)]) 
    
    D_k_penrose = np.linalg.pinv(reconstructed_D) 
    A_bi_cross_estimate = np.dot(np.dot(B.transpose(),D_k_penrose.transpose()),C.transpose())
    
    return A_bi_cross_estimate.transpose()
    
def score(*args,**kwargs):
    
    options = {
    'k' : 2,
    'd_n_rows' : int(args[0].shape[0]/2),
    'd_n_cols' : int(args[0].shape[1]/2),
    'SVD_package'  : 'numpy'}
    
    options.update(kwargs)

    A_bi_cross_estimate = predict(*args,**kwargs)
    A,B,C,D = break_X_into_ABCD(args[0],options['d_n_rows'],options['d_n_cols'])
    return np.sum((A-A_bi_cross_estimate)**2)

##################################################################
##################################################################
######Stuff below here is more optimzed for speed ################
##################################################################
##################################################################


def score_fast(y_stack,**kwargs):


    score_list = []

    for i in np.arange(kwargs['n_iterations']):

        update_progress((i) / kwargs['n_iterations'])

        my_prediction = predict_vs_k(y_stack,**kwargs)

        k,scores = np.array([i for i in my_prediction]).transpose()
        score_list.append(scores)
    update_progress(1)
    return k,np.array(score_list)

#this is a fast version
def predict_vs_k(y_stack,**kwargs):
    options = {
        'k' : 2,
        'd_n_rows'     : int(y_stack.shape[0]/2),
        'd_n_cols'     : int(y_stack.shape[1]/2),
        'k_list'       : np.arange(1,int(np.min(y_stack.shape)/4))}

    options.update(kwargs)

    A,B,C,D = break_X_into_ABCD(y_stack,options['d_n_rows'],options['d_n_cols'])
    
        
    Ud,Sd,Vd = np.linalg.svd(D,full_matrices=False)

    updated_k_list = np.sort(options['k_list'])[::-1]

    for i in updated_k_list:

        
        Sd[i:] = 0
        reconstructed_D = np.dot(np.dot(Ud,np.diag(Sd)),Vd[:len(Sd)]) 
        
        D_k_penrose = np.linalg.pinv(reconstructed_D) 
        A_bi_cross_estimate = (np.dot(np.dot(B.transpose(),D_k_penrose.transpose()),C.transpose())).transpose()
        score = np.sum((A-A_bi_cross_estimate)**2)
        yield(np.array([i,score]))
    
    #return A_bi_cross_estimate.transpose()
