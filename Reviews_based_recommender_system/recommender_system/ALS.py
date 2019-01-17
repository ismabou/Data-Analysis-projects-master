import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

def train_and_test(R):
    
    """random_index is an array of booleans with the same length as the number of non-zero ratings in R
    Each value has a probability of 0.9 to be True.
    Then, for each non-zero element of R, we add it into R_train if its index in random_index is True,
    otherwise we add it into R_test (we inverted the boolean list random_index to take the appropriate elements
    in R_test). The other values of both matrix are 0"""
    
    index_users = R.nonzero()[0]
    index_movies = R.nonzero()[1]
    predictions = R[index_users, index_movies]
    
    random_index = np.random.choice([False,True], p=[0.1,0.9], size=index_users.shape[0])
    R_train = np.zeros(R.shape)
    R_test = np.zeros(R.shape)

    R_train[index_users[random_index],index_movies[random_index]] = predictions[random_index]
    R_test[index_users[~random_index],index_movies[~random_index]] = predictions[~random_index]

    
    for r in range(R_train.shape[0]):
        if (np.sum(R_train[r])==0):
            R_train[r]=R[r]
    for c in range(R_train.shape[1]):
        if (np.sum(R_train[:,c])==0):
            R_train[:,c]=R[:,c]
    
    return R_train , R_test 


def rmse(R,R_pred,I):    
    
    return np.sqrt(mse(R[I[0],I[1]],R_pred[I[0],I[1]]))

def init_M(R,n_f):
    
    """This function returns the initial value of the matrix M as stated in the report.
    M is an n_f x n_m matrix. 
    The first feature of each movie is its average rating over the users.
    The average is computed over the available ratings so the ones different than 0.  
    The other fields of the matrix are small random numbers between 0 and 1"""
    
    averages = np.array([np.mean(R[:,i][R[:,i] != 0]) for i in range(R.shape[1])]).reshape([1,R.shape[1]])
    M = np.r_[averages,np.random.rand(n_f - 1,R.shape[1])/10]
    return M

def calculus(X,l,axis,R):
    
    """This function returns two lists. The first one is I. Each element of I corresponds to a list of integers representing
    the columns that we have to select in the matrix X so the second one is composed of the sub-matrixs taken with respect
    to the set I_i.
    The reason we have this function is that both update_M and update_U functions have to select some columns 
    for their matrix so to modularize the code, we created this function.
    For the parameter axis, it can be either [1,0] or [0,1]. It selects the right set to choose for U and M.
    index is a matrix of 2 columns composed of the indexs where there is a rating different than 0.
    
    
    
    
    
    index[axis[0]] corresponds to the column of the dataframe df. index[axis[1]] corresponds to the index of df.
    Be aware that df has only one column named '0' by default.
    We used a dataframe because it is easy for example to have all the movies rated by user i (df.loc[i], if the
    index of df are the users' index, would return all the lines with index i)
    Each element of I denoted I_i is a list of integers taken from the column of df and with index i. 
    The for loop is over R.shape[axis[1]] so over the data frame index.

    For example if axis is [0,1] then df would have as index the movies and as column the users. Each line
    corresponds to the index (u,m) of a user u and a movie m where R[u,m] is different than 0.
    I_i would be all the users' index that have rated the movie i"""
    
    index = R.nonzero()
    
    df = pd.DataFrame(index[axis[0]],index[axis[1]])
    I=[]
    for i in range(R.shape[axis[1]]):
            
            spl=df.loc[i][0]
        
            try:
                I.append(spl.values)
            except:
                I.append(np.array([spl]))
                pass
        
    
 #   I = [df.loc[i][0].values for i in range(R.shape[axis[1]])]
    
    
    """Each element of X_I corresponds to the selected columns of X. Those selected columns are the elements in each I_i.
    For example, the first element of I is a list of integers. Thus, the first element of X_I is composed of the
    I_1 columns of X"""
    
    X_I = [X[:,i] for i in I]
    return I,X_I

def update_U(M,l,R):    
    
    
    """This function udpate U with respect to the algorithm in the report
    Each element I_i of I corresponds to a list of integers representing the columns that we have to select in M
    Since axis = [1,0] then the set I_i is just all the movies' index that have been rated by user i.
    For example I_i can be [0,1,8,...,999]
    Each element of M_I corresponds to the matrix M_(I_i) which is the sub-matrix with columns selected with
    the corresponding element of I (which is the list of integers I_i)
    Each element of R_I corresponds to the vector R(i,I_I) which is composed of the I_i columns of the i-th row of R
    We return the matrix U. It has the same number of the rows than R (R.shape[0] so n_u). Plus, each columns of U 
    corresponds to the formula stated on the report. At the end, we will have a matrix n_u x n_f 
    so we have to transpose it."""
    
    I, M_I = calculus(M,l,[1,0],R)
    R_I = [R[i,I[i]] for i in range(len(I))]
    return np.array([np.dot(np.linalg.inv(np.dot(M_I[i],M_I[i].T) + l*I[i].shape[0]*\
                                          np.eye(M.shape[0])),np.dot(M_I[i],R_I[i])) for i in range(R.shape[0])]).T
    
def update_M(U,l,R):
    
    
    """This function update M with respect to the algorithm in the report
    Each element I_i of I corresponds to a list of integers caracterizing the columns that we have to select in U
    Since axis = [0,1] then the set I_i is just all the users' index that rated movie i.
    For example I_i can be [0,1,8,...,9999]
    Each element of U_I corresponds to the matrix U_(I_i) which is the sub-matrix with columns selected with
    the corresponding element of I (which is the list of integers I_i)
    Each element of R_I corresponds to the vector R(i,I_I) which is composed of the I_i rows of the i-th column of R
    We return the matrix M. It has the same column number than R (R.shape[1] so n_m). Plus, each columns of U corresponds 
    to the formula stated on the report. At the end, we will have a matrix n_m x n_f so we have to transpose it."""
    
    I, U_I = calculus(U,l,[0,1],R)
    
    R_I = [R[I[i],i] for i in range(len(I))]
    return np.array([np.dot(np.linalg.inv(np.dot(U_I[i],U_I[i].T) + l*I[i].shape[0]*\
                                          np.eye(U.shape[0])),np.dot(U_I[i],R_I[i])) for i in range(R.shape[1])]).T
    
def ALS_WR(R,Rtest,n_f,l,nb_iter_max,epsilon):
    
    """This is the main algorithm. We initialize M then the rmse_old to +infinity since we want to minimize it.
    We repeat each time the algorithm until the difference between the current and the old rmse is less than the
    stopping criteria or we have made a maximum number of iterations."""
    rmses=[]
    M = init_M(R,n_f)
    nb_iter = 0
    rmse_old = np.inf
    
    I = Rtest.nonzero()    
    
    while nb_iter < nb_iter_max:
        
        nb_iter += 1
        #print("iteration: "+str(nb_iter)+" rmse : "+str(rmse_old))
        
        U = update_U(M,l,R)
        M = update_M(U,l,R)
        
        R_pred = np.dot(U.T,M)       

        current_rmse = rmse(Rtest,R_pred,I)        
        if abs(current_rmse - rmse_old) < epsilon:
            return current_rmse, R_pred,rmses
        rmse_old = current_rmse
        rmses.append(rmse_old)
      
    return R_pred,rmses


def compute_bias(ratings_matrix,n_f,l,nb_iter_max,epsilon) :
    
    """Compute the global bias of the predictions (as explained in the report) with the same parameters used for ALS.
    First we split the matrix into a train matrix and a test matrix, compute predictions on the train matrix, then calculate the bias using     the known ratings of the test matrix""" 
    
    train , test = train_and_test(ratings_matrix)   
    R_pred_test = ALS_WR(train,n_f,l,nb_iter_max,epsilon)
    
    I2 = test.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0
    
    bias = (test - I2*R_pred_test).sum()/(I2.sum())
   
    return  bias

