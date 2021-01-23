import numpy as np
from scipy.interpolate import RectBivariateSpline
import time

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: Current image
    :param It1: Next image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """
    start1 = time.time()
    # put your implementation here
    M = np.eye(3) #one on the diagonal
    p = np.zeros(6)
    print("Lucas Aff Called")
    
      #X is numb of cols -> horizontal
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    r_row, r_col = y2-y1, x2-x1
    r_row, r_col = int(r_row), int(r_col)
    
    # 1 Warping rect box to image & template
    #Make rect in coords to warp in template
    col_count = np.linspace(x1, x2, r_col) 
    row_count = np.linspace(y1, y2, r_row)
    c_coord, r_coord = np.meshgrid(col_count, row_count)
    
    #Warp the rect in template image
    empt_col = np.arange(0, It.shape[0], 1); empt_row = np.arange(0, It.shape[1], 1) #axis=0 is thru a col; ydir
    spln_temp = RectBivariateSpline(empt_col, empt_row, It)
    rect_temp = spln_temp.ev(r_coord, c_coord) #may be upside down
    # print("rect_temp shape: ", rect_temp.shape)
    
    
    #3a Gradient descent -> Create spline template for smoothing in each new rect 
    der_y, der_x = np.gradient(It1) 
    spln_dx = RectBivariateSpline(empt_col, empt_row, der_x)
    spln_dy = RectBivariateSpline(empt_col, empt_row, der_y)
    
    #4 Jacobian of dw/dp -> inside loop coz need x & y
    
    spln_img = RectBivariateSpline(empt_col, empt_row, It1)
    end1 = time.time()
    print("Outside loop time: ", end1 - start1)
    for i in range(int(num_iters)):
        print(i)
        start2 = time.time()
        #2 Compute error image (Real image - warped template)
        x1m = x1*M[0,0] + y1*M[0,1] + M[0,2]; y1m = x1*M[1,0] + y1*M[1,1] + M[1,2]
        x2m = x2*M[0,0] + y2*M[0,1] + M[0,2]; y2m = x2*M[1,0] + y2*M[1,1] + M[1,2]
        col_count_p = np.linspace(x1m, x2m, It.shape[1]) #linear interpolation
        row_count_p = np.linspace(y1m, y2m, It.shape[0])
        c_coord_p, r_coord_p = np.meshgrid(col_count_p, row_count_p) #updated coords
        
        #Interpolate new area to original image
        rect_img = spln_img.ev(r_coord_p, c_coord_p)
        # print("rect_img shape: ", rect_img.shape)
        
        #error
        Err = rect_temp - rect_img
        Err = Err.reshape(-1,1) # nx1
        
        #3b
        g_dx = spln_dx.ev(r_coord_p, c_coord_p)
        g_dy = spln_dy.ev(r_coord_p, c_coord_p)
        delI = np.vstack((np.ravel(g_dx), np.ravel(g_dy))) # 2xn
        delI = np.transpose(delI) # nx2
        
        #5 Hessian
        GJ = np.zeros((It.shape[0] * It.shape[1], 6)) #nx2 * 2x6 = nx6
        for x in range(It.shape[0]):
            for y in range(It.shape[1]):
                #For each pixel, delI: 1x2 & Jac: 2x6. n -> all pixels
                delI_ind = np.array([delI[x * It.shape[1] + y]]).reshape(1,2)
                Jac_ind = np.array([[y, 0, x, 0, 1, 0],
                                    [0, y, 0, x, 0, 1]]) 
                GJ[x * It.shape[1] + y] = delI_ind @ Jac_ind #GJ: nx6
        
        hess = np.dot(np.transpose(GJ), GJ) #hess: 6x6
        hess_inv = np.linalg.inv(hess)
        
        #6 Del P
        delp = np.dot(hess_inv, np.transpose(GJ))
        delp = np.dot(delp, Err) #should be 6x1
        
        #7 Updating M or p
        print(delp)
        p[0] += delp[0,0]; p[1] = delp[1,0]; p[2] = delp[2,0]
        p[3] += delp[3,0]; p[4] = delp[4,0]; p[5] = delp[5,0]
        
        M[0,0] = 1.0 + p[0]; M[0,1] = p[1]; M[0,2] = p[2]
        M[1,0] = p[3]; M[1,1] = 1.0 + p[4]; M[1,2] = p[5]
        
        check = delp**2
        sumsq = np.sum(check)
        print(sumsq)
        end2 = time.time()
        print("Inside per loop time: ", end2 - start2)
        if (sumsq < threshold):
            print("Lucas sumsq done")
            return M

    print("Lucas done")
    return M
