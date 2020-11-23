import tensorflow as tf
import numpy as np


# (L,H,W) is the size of filters in wc layer
def Aproxi_tensor(i,wc1,wc2):
#    wcc=wc
    m = wc1[0, 0, 0, :].size
    n = wc1[:, :, :, 0].size
    f = wc1[:, :, :, 0].shape
    posi_layer_ind = position_vec(wc2)
    matrix1 = filt_mat_from(wc1, m, n)
    MatOfTen = matrix1
#    wc = wcc
    DelRow=[]
    for s in range(i):
        DelRow.append(posi_layer_ind[s])
    DelRow.sort(reverse = True)
    for r in range(i):
        MatOfTen=np.delete(MatOfTen, DelRow[r], axis=0)
    for j in range(i):
        Bestvec = best_appr(MatOfTen,matrix1[posi_layer_ind[j],:])
        wc1[:,:,:,posi_layer_ind[j]] = Bestvec.reshape(f)
    return wc1

# Matrix Form of a Filter
def filt_mat_from(x, k, l):
    matr_form = np.zeros((k, l))
    for i in range(k):
        row_x = x[:, :, :, i]
        matr_form[i, :] = row_x.reshape(1, l)
    return matr_form


# Beta vector
def beta_vec(x,y):
    Beta=np.matmul(x,y)
    return Beta


# G Matrix
def coff_g_mat(x):
    g_coff=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        g_coff[:,i]=np.matmul(x,x[i,:])
    return g_coff


# Best approximation of kth row in the rest of row space of matrix
def best_appr(x,y):
    G = coff_g_mat(x)
    beta = beta_vec(x,y)
    alpha = np.matmul(np.linalg.inv(G), beta)
    best = np.matmul(np.transpose(alpha), x)
    return best


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def position_vec (wc):
    m = wc[0,0,0,:].size
    n = wc[0,0,:,0].size
    layer_ind = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            x1 = wc[:, :, j, i].reshape(1, 9)
            layer_ind[i, j] = np.matmul(x1, np.transpose(x1))
    layer_ind_3 = np.transpose(np.matmul(np.ones((1, m)), layer_ind))
    ordered_layer_ind = layer_ind_3
    posi_layer_ind = []
    Max_3 = np.max(ordered_layer_ind)
    for i in range(len(ordered_layer_ind)):
        posi_layer_ind.append(np.argmin(ordered_layer_ind))
        ordered_layer_ind[np.argmin(ordered_layer_ind)] = abs(layer_ind_3[np.argmin(ordered_layer_ind)]) + abs(
            Max_3)
    return posi_layer_ind

def channel_mul(t,r,s,A,B): #channel convolution for thinet(A: output, B: filter)
    #with tf.Session() as sess:
     #   C = sess.run(A)
    out_cha_nu = A[0,0,0,:].size
    fil_row_nu = B[:,0,0,0].size
    fil_colu_nu = B[0,:,0,0].size
    cha_mul = np.zeros((1,out_cha_nu))
    for i in range(out_cha_nu):
        C_cel = A[0,(t*fil_row_nu):(((t+1)*(fil_row_nu))),(r*fil_colu_nu):((r+1)*(fil_colu_nu)),i].reshape(1,(fil_row_nu)*(fil_row_nu))
        C_vec = B[:,:,i,s].reshape((fil_row_nu)*(fil_row_nu),1)
        cha_mul[0,i]= np.matmul(C_cel,C_vec)
    return cha_mul

def filter_selec(X,A,B): #X: the array of the entries of the output, A: output, B: filter
    with tf.Session() as sess:
        C = sess.run(A)
    out_cha_nu = C[0,0,0,:].size
    Del = np.zeros((len(X),out_cha_nu))
    for i in range(len(X)):
        Del[i,:]= channel_mul((X[i])[0],(X[i])[1],(X[i])[2],C,B)**2
    return Del

def arrange(A):
    Yek = np.ones((1, len(A)))
    channel_ind = np.transpose(np.matmul(Yek,A))
    ordered_layer_ind = channel_ind
    posi_layer_ind = []
    for i in range(len(channel_ind)):
        D_ORDER = np.argmin(ordered_layer_ind)
        posi_layer_ind.append(D_ORDER)
        ordered_layer_ind = np.delete(ordered_layer_ind, D_ORDER)
    return posi_layer_ind