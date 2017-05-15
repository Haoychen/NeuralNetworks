import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image

import theano
import theano.tensor as T


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
    '''
    This function reconstructs an image X_recon_img given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
        

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]
    
    #TODO: Enter code below for reconstructing the image X_recon_img
    X_mean_xsize = X_mean.shape[0]
    X_mean_ysize = X_mean.shape[1]
    X_mean = X_mean.reshape(X_mean_xsize * X_mean_ysize)
    X_recon_img = np.dot(c_im.T, D_im.T) + X_mean
    X_recon_img.resize(256, 256)
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num), cmap='Greys_r')
            
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them
    f, axarr = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            plt.imshow(D[:,i*4 + j].reshape(sz, sz), cmap='Greys_r')
    f.savefig(imname)
    plt.close(f)

    #raise NotImplementedError

    
def main():
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Read all images into a numpy array of size (no_images, height, width)
    path = 'Fei_256/'
    images = os.listdir(path)
    images = sorted(images)

    no_images = 200
    images_ndarray = np.zeros((no_images, 256, 256))
    size = (256, 256)
    i = 0
    for image in images:
        if 'jpg' in image:
            im = Image.open(path + image).convert('L')
            greyscale_map = np.asarray(im)
            images_ndarray[i] = greyscale_map.reshape(size)
            i += 1


    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        ''' 
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        n_block = 256 / sz
        n_blocks_in_image = n_block * n_block
        X = np.zeros((no_images * n_blocks_in_image, sz * sz))

        for i in range(no_images):
            img = images_ndarray[i]
            img_div = [img[m * sz:(m + 1) * sz, n * sz:(n + 1) * sz].flatten() for m in range(n_block) for
                               n in range(n_block)]
            for j in range(n_blocks_in_image):
                X[(i * n_blocks_in_image) + j] = img_div[j].copy()



        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        hat_matrix = np.dot(X.transpose(), X)
        eigenvalues, eigenvectors = np.linalg.eigh(hat_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        D = eigenvectors



        c = np.dot(D.T, X.T)
        
        for i in range(0, 200, 10):
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256/sz), im_num=i)

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
    
    