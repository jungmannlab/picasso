import numpy as _np
cimport numpy as np
ctypedef np.float64_t DTYPE_t
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

def cfill_gaussian_rot(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] z,
     np.ndarray[float, ndim=1] sx, np.ndarray[float, ndim=1] sy, np.ndarray[float, ndim=1] sz,
      int n_pixel_x, int n_pixel_y, (float, float) ang):
    
    cdef float _DRAW_MAX_SIGMA = 3.0
    cdef np.ndarray[DTYPE_t, ndim=2] image = _np.zeros((n_pixel_y, n_pixel_x))
    cdef float x_, y_, z_, sx_, sy_, sz_ 
    cdef int count = 0

    cdef float max_y, max_x, max_z
    cdef int i_min, i_max, j_min, j_max, k_min, k_max
    cdef np.ndarray[double, ndim=2] cov_matrix, rot_mat_x, rot_mat_y, rot_matrix, cov_rot, cri
    cdef int i, j, k
    cdef float a, b, c, exponent

    cdef float angx = ang[0]
    cdef float angy = ang[1]

    rot_mat_x = _np.array([[1.0,0.0,0.0],[0.0,_np.cos(angx),_np.sin(angx)],[0.0,-_np.sin(angx), _np.cos(angx)]])
    rot_mat_y = _np.array([[_np.cos(angy),0.0,_np.sin(angy)],[0.0,1.0,0.0],[-_np.sin(angy),0.0,_np.cos(angy)]])
    rot_matrix = mat_mul(rot_mat_x, rot_mat_y)
    cdef np.ndarray[double, ndim=2] rot_matrixT = rot_matrix.T

    for x_, y_, z_, sx_, sy_, sz_ in zip(x, y, z, sx, sy, sz):
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = int(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = int(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = int(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = int(x_ + max_x + 1)
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        max_z = _DRAW_MAX_SIGMA * sz_
        k_min = int(z_ - max_z)
        k_max = int(z_ + max_z + 1)

        cov_matrix = _np.array([[sx_**2, 0, 0], [0, sy_**2, 0], [0, 0, sz_**2]]) #covariance matrix 
        cov_rot = mat_mul(rot_matrixT, mat_mul(cov_matrix, rot_matrix))
        cri = inverse(cov_rot) #stands for covariance rotated inverse
        count += 1
        if count % 50 == 0:
            print(count)

        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                a = j + 0.5 - x_
                b = i + 0.5 - y_
                for k in range(k_min, k_max):   
                    c = k + 0.5 - z_
                    exponent = a*a * cri[0,0] + a*b * cri[0,1] + a*c * cri[0,2] + \
                               a*b * cri[1,0] + b*b * cri[1,1] + b*c * cri[1,2] + \
                               a*c * cri[2,0] + b*c * cri[2,1] + c*c * cri[2,2]
                    image[i,j] += 2.71828 ** (-0.5 * exponent) / ((6.28319**3 * determinant(cov_rot)) ** 0.5)
    return image

cdef double determinant(np.ndarray[double, ndim=2] s):
    # s is assumed to have shape 3x3
    return s[0,0] * (s[1,1] * s[2,2] - s[1,2] * s[2,1]) - \
        s[0,1] * (s[1,0] * s[2,2] - s[2,0] * s[1,2]) + \
        s[0,2] * (s[1,0] * s[2,1] - s[2,0] * s[1,1])

cdef np.ndarray[double, ndim=2] mat_mul(np.ndarray[double, ndim=2] a, np.ndarray[double, ndim=2] b):
    # both a and b are assumend to be of shape 3x3
    cdef int i, j, k
    cdef np.ndarray[double, ndim=2] c = _np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i,j] += a[i,k] * b[k,j]
    return c

cdef np.ndarray[double, ndim=2] inverse(np.ndarray[double, ndim=2] a):
    #again, a is a 3x3 matrix
    cdef np.ndarray[double, ndim=2] c = _np.zeros((3,3))
    cdef double det = determinant(a)

    c[0,0] = (a[1,1] * a[2,2] - a[1,2] * a[2,1]) / det
    c[0,1] = (a[0,2] * a[2,1] - a[0,1] * a[2,2]) / det
    c[0,2] = (a[0,1] * a[1,2] - a[0,2] * a[1,1]) / det

    c[1,0] = (a[1,2] * a[2,0] - a[1,0] * a[2,2]) / det
    c[1,1] = (a[0,0] * a[2,2] - a[0,2] * a[2,0]) / det
    c[1,2] = (a[0,2] * a[1,0] - a[0,0] * a[1,2]) / det

    c[2,0] = (a[1,0] * a[2,1] - a[1,1] * a[2,0]) / det
    c[2,1] = (a[0,1] * a[2,0] - a[0,0] * a[2,1]) / det
    c[2,2] = (a[0,0] * a[1,1] - a[0,1] * a[1,0]) / det

    return c
