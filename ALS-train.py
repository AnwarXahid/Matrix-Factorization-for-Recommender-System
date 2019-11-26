import numpy as np
import pandas as pd
import random as rd
import xlrd
import sys
import math

rd.seed(17)


############ update Item_Mat using User_Mat
def update_v_with_u(x, u, v):
    for i in range(M):
        AccMat1 = np.zeros((K, K))
        AccMat2 = np.zeros(K)
        for j in range(N):
            if x[j][i] != -1:
                temp = u[j, :]
                temping = [temp]
                #print temping
                tempTrans = np.transpose(temping)
                #print tempTrans
                tempMul = np.matmul(tempTrans, temping)
                #print tempMul
                tempIden = lam_v * np.identity(K)
                #print tempIden
                tempAdd = np.add(tempMul, tempIden)
                #print tempAdd
                AccMat1 = np.add(AccMat1, tempAdd)
                #print AccMat1
                AccMat2 = np.add(AccMat2, x[j][i] * temp)
                #print AccMat2
                #print np.linalg.det(AccMat1)
        updatedColumn = np.matmul(np.linalg.inv(AccMat1), AccMat2)
        #print "up"
        #print updatedColumn
        for l in range(K):
            v[l][i] = updatedColumn[l]
        #print v



############ update User_Mat using Item_Mat
def update_u_with_v(x, u, v):
    for i in range(N):
        AccMat1 = np.zeros((K, K))
        AccMat2 = np.zeros(K)
        for j in range(M):
            if x[i][j] != -1:
                temp = v[:, j]
                temping = [temp]
                tempTrans = np.transpose(temping)
                tempMul = np.matmul(tempTrans, temping)
                #print tempMul
                tempIden = lam_u * np.identity(K)
                tempAdd = np.add(tempMul, tempIden)
                AccMat1 = np.add(AccMat1, tempAdd)
                AccMat2 = np.add(AccMat2, x[i][j] * temp)
                #print np.linalg.det(AccMat1)
        updatedColumn = np.matmul(np.linalg.inv(AccMat1), AccMat2)
        for l in range(K):
            u[i][l] = updatedColumn[l]


def calculate_loss(x, u, v):
    term_1 = 0
    term_2 = 0
    term_3 = 0
    for i in range(N):
        for j in range(M):
            if x[i][j] != -1 :
                temp_u = u[i, :]
                temp_v = v[:, j]
                temp = x[i][j] - np.matmul(temp_u, temp_v)
                term_1 += temp * temp

    for i in range(N):
        temp = 0
        for j in range(K):
            temp += u[i][j] * u[i][j]
        term_2 += lam_u * math.sqrt(temp)

    for i in range(M):
        temp = 0
        for j in range(K):
            temp += v[j][i] * v[j][i]
        term_3 += lam_v * math.sqrt(temp)

    return term_1 + term_2 + term_3



def calculate_RMSE(x, p):
    loss = 0
    dim = 0
    for i in range(N):
        for j in range(M):
            if x[i][j] != -1 :
                loss += ( x[i][j] - p[i][j] ) ** 2
                dim += 1
    return math.sqrt(loss / dim)


"""
############### Hyperparameter ###################
K = 10
lam_u = 0.01
lam_v = 0.01
##################################################

DataFrame = pd.read_excel("ratings_train.xlsx", sheet_name=0, header= None)
train_X = np.array(DataFrame._values)
N, M = np.array(train_X).shape

train_U = np.random.uniform(0, 5, [N, K])
train_V = np.random.uniform(0, 5, [K, M])




#################          main algorithm          #####################
prev_loss = 0
curr_loss = 200
while abs(prev_loss - curr_loss) > 100 :
    prev_loss = curr_loss
    update_v_with_u(train_X, train_U, train_V)
    update_u_with_v(train_X, train_U, train_V)
    curr_loss = calculate_loss(train_X, train_U, train_V)
    #print User_Mat

print calculate_RMSE(train_X, np.matmul(train_U, train_V))

################# end of training #####################

#################### start validation #################

df = pd.read_excel("ratings_validate.xlsx", sheet_name=0, header= None)
validate_X = np.array(df._values)
N, M = np.array(validate_X).shape

validate_U = np.random.uniform(0, 5, [N, K])
update_u_with_v(validate_X, validate_U, train_V)
print calculate_RMSE(validate_X, np.matmul(validate_U, train_V))
################end of validation ################


############### start of testing #################




############# end of testing ####################
"""






K_array = [10, 20, 40]
lamda_u_array = [0.01, 0.1, 1, 10]
min_RMSE = sys.maxint
test_K = 10
test_lamda = 0.01

DataFrame = pd.read_excel("ratings_train.xlsx", sheet_name=0, header= None)
train_X = np.array(DataFrame._values)

df = pd.read_excel("ratings_validate.xlsx", sheet_name=0, header= None)
validate_X = np.array(df._values)



for i in range(3):
    for j in range(4):
        K = K_array[i]
        lam_u = lamda_u_array[j]
        lam_v = lam_u
        N, M = np.array(train_X).shape

        train_U = np.random.uniform(0, 5, [N, K])
        train_V = np.random.uniform(0, 5, [K, M])

        prev_loss = 0
        curr_loss = 200
        while abs(prev_loss - curr_loss) > 100:
            prev_loss = curr_loss
            update_v_with_u(train_X, train_U, train_V)
            update_u_with_v(train_X, train_U, train_V)
            curr_loss = calculate_loss(train_X, train_U, train_V)

        print("Traing RMSE for K = ", K, ", lamda_u = ", lam_u, " : ")
        print calculate_RMSE(train_X, np.matmul(train_U, train_V))

        N, M = np.array(validate_X).shape

        validate_U = np.random.uniform(0, 5, [N, K])
        update_u_with_v(validate_X, validate_U, train_V)
        print("validation RMSE for K = ", K, ", lamda_u = ", lam_u, " : ")
        val_RMSE = calculate_RMSE(validate_X, np.matmul(validate_U, train_V))
        print val_RMSE
"""
        if val_RMSE < min_RMSE:
            min_RMSE = val_RMSE
            test_K = K
            test_lamda = lam_u
            trained_v = train_V


K = test_K
lam_u = test_lamda
lam_v = test_lamda

df_test = pd.read_excel("ratings_test.xlsx", sheet_name=0, header= None)
test_X = np.array(df_test._values)
N, M = np.array(test_X).shape

test_U = np.random.uniform(0, 5, [N, K])
update_u_with_v(test_X, test_U, trained_v)
print calculate_RMSE(validate_X, np.matmul(validate_U, trained_v))
"""

