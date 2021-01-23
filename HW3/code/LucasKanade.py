import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p = p0

    print("Called", p)

    
    #X is numb of cols -> horizontal
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
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
    
    #4 Jacobian of dw/dp
    jacob = np.asarray([[1, 0],[0, 1]]) #Translation -> Identity
    
    spln_img = RectBivariateSpline(empt_col, empt_row, It1)
        
    for i in range(int(num_iters)):
        #2 Compute error image (Real image - warped template)
        #Make rect in new coords (after p) to warp in template
        x1p, y1p, x2p, y2p = x1+p[0], y1+p[1], x2+p[0], y2+p[1]
        col_count_p = np.linspace(x1p, x2p, r_col) #linear interpolation
        row_count_p = np.linspace(y1p, y2p, r_row)
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
        GJ = np.dot(delI, jacob)
        hess = np.dot(np.transpose(GJ), GJ) #nx2
        hess_inv = np.linalg.inv(hess)
        
        #6 Del P
        delp = np.dot(hess_inv, np.transpose(GJ))
        delp = np.dot(delp, Err) #should be 2x1
        
        #7 Updating p
        p = p + delp[:,0]
        # print("Loop ", i)
        # print("p is ", p)
        
        check = delp**2
        sumsq = np.sum(check)
        if (sumsq < threshold):

            return p
  
    return p

# w_rect = np.asarray([rect[0]+p[0] , rect[1]+p[1] , rect[2]+p[0] , rect[3]+p[1]])