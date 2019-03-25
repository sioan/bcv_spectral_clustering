import numpy as np

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
