import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import affine_transform, convolve, correlate

MOTION_TRANSLATION = 0
MOTION_EUCLIDEAN = 1
MOTION_AFFINE = 2
MOTION_HOMOGRAPHY = 3

MOTION_TYPES = ('MOTION_AFFINE', 'MOTION_TRANSLATION', 'MOTION_HOMOGRAPHY', 'MOTION_EUCLIDEAN')
MOTION_TYPES_CV = (0, 1, 2, 3)
MOTION_TYPES_DICT = {
    'MOTION_TRANSLATION': 0,
    'MOTION_EUCLIDEAN': 1,
    'MOTION_AFFINE': 2,
    'MOTION_HOMOGRAPHY': 3
}

ParamTEMP = {
    0: 2,
    1: 3,
    2: 6,
    3: 8
}


class CriteriaECC(object):
    def __init__(self, num_iter=10000, epsilon=1e-4) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.eps = epsilon


def inv_warp_mat(mat):
    assert len(mat.shape) == 2
    assert mat.shape[1] == 3
    if isinstance(mat, np.ndarray):

        if mat.shape[0] == 2:
            reverse_mat = np.zeros_like(mat)
            inv_22 = np.linalg.inv(mat[:2, :2])
            reverse_mat[:2, :2] = inv_22
            reverse_mat[:2, 2] = -1 * np.matmul(inv_22, mat[:, 2])
        elif mat.shape[0] == 3:
            reverse_mat = np.linalg.inv(mat)
        else:
            raise RuntimeError('Warp_matrix shape error.')
    elif isinstance(mat, cp.ndarray):

        if mat.shape[0] == 2:
            reverse_mat = cp.zeros_like(mat)
            inv_22 = cp.linalg.inv(mat[:2, :2])
            reverse_mat[:2, :2] = inv_22
            reverse_mat[:2, 2] = -1 * cp.matmul(inv_22, mat[:, 2])
        elif mat.shape[0] == 3:
            reverse_mat = cp.linalg.inv(mat)
        else:
            raise RuntimeError('Warp_matrix shape error.')
    else:
        raise NotImplementedError('inv_warp_mat supports matrix in shape (2,3) or (3,3)')
    return reverse_mat


def convert_mat_sci(mat, inverse=True):
    if isinstance(mat, np.ndarray):
        sci_mat = np.array([[mat[1, 1], mat[1, 0], mat[1, 2]], [mat[0, 1], mat[0, 0], mat[0, 2]]]).reshape(2, 3)
    elif isinstance(mat, cp.ndarray):
        sci_mat = cp.zeros_like(mat)
        sci_mat[0, 0] = mat[1, 1]
        sci_mat[0, 1] = mat[1, 0]
        sci_mat[0, 2] = mat[1, 2]
        sci_mat[1, 0] = mat[0, 1]
        sci_mat[1, 1] = mat[0, 0]
        sci_mat[1, 2] = mat[0, 2]
        # sci_mat = cp.array([[mat[1, 1], mat[1, 0], mat[1, 2]], [mat[0, 1], mat[0, 0], mat[0, 2]]]).reshape(2, 3)
    else:
        raise NotImplementedError
    if inverse:
        sci_mat = inv_warp_mat(sci_mat)
    return sci_mat


def image_jacobian_euclidean_ECC(jacobian, gradientX_warped, gradientY_warped, x_grid, y_grid, warp_matrix):
    rows, cols = gradientY_warped.shape

    h0 = warp_matrix[0, 0]
    h1 = warp_matrix[1, 0]

    hat_x = -1 * x_grid * h1 - y_grid * h0
    hat_y = x_grid * h0 - y_grid * h1

    jacobian[:, :cols] = gradientX_warped * hat_x + gradientY_warped * hat_y
    jacobian[:, cols:2 * cols] = gradientX_warped
    jacobian[:, 2 * cols:3 * cols] = gradientY_warped

    return jacobian


def image_jacobian_homo_ECC(jacobian, gradientXWarped, gradientYWarped, Xgrid, Ygrid, warp_matrix):
    h0_ = warp_matrix[0, 0]
    h1_ = warp_matrix[1, 0]
    h2_ = warp_matrix[2, 0]
    h3_ = warp_matrix[0, 1]
    h4_ = warp_matrix[1, 1]
    h5_ = warp_matrix[2, 1]
    h6_ = warp_matrix[0, 2]
    h7_ = warp_matrix[1, 2]

    w = gradientXWarped.shape[1]

    den_ = Xgrid * h2_ + Ygrid * h5_ + 1.0

    hatX_ = -1 * Xgrid * h0_ - Ygrid * h3_ - h6_
    hatX_ = hatX_ / den_

    hatY_ = -1 * Xgrid * h1_ - Ygrid * h4_ - h7_
    hatY_ = hatY_ / den_

    gradientXWarped_div = gradientXWarped / den_
    gradientYWarped_div = gradientYWarped / den_

    jacobian[:, :w] = gradientXWarped_div * Xgrid
    jacobian[:, w:2 * w] = gradientYWarped_div * Xgrid

    temp_ = hatX_ * gradientXWarped_div + hatY_ * gradientYWarped_div
    jacobian[:, 2 * w:3 * w] = temp_ * Xgrid

    jacobian[:, 3 * w:4 * w] = gradientXWarped_div * Ygrid
    jacobian[:, 4 * w:5 * w] = gradientYWarped_div * Ygrid
    jacobian[:, 5 * w:6 * w] = temp_ * Ygrid
    jacobian[:, 6 * w:7 * w] = gradientXWarped_div
    jacobian[:, 7 * w:8 * w] = gradientYWarped_div
    return jacobian


def image_jacobian_affine_ECC(jacobian, gradientXWarped, gradientYWarped, Xgrid, Ygrid):
    w = gradientXWarped.shape[1]

    jacobian[:, :w] = gradientXWarped * Xgrid
    jacobian[:, w:2 * w] = gradientYWarped * Xgrid
    jacobian[:, 2 * w:3 * w] = gradientXWarped * Ygrid
    jacobian[:, 3 * w:4 * w] = gradientYWarped * Ygrid

    jacobian[:, 4 * w:5 * w] = gradientXWarped
    jacobian[:, 5 * w:6 * w] = gradientYWarped

    return jacobian


def image_jacobian_translation_ECC(jacobian, gradientXWarped, gradientYWarped):
    w = gradientXWarped.shape[1]

    jacobian[:, :w] = gradientXWarped
    jacobian[:, w:2 * w] = gradientYWarped
    return jacobian


def project_onto_jacobian_ECC(src1, src2, dst):
    src1_h, src1_w = src1.shape
    src2_h, src2_w = src2.shape
    assert src1_h == src2_h

    if not src1_w == src2_w:
        w = src2_w
        assert dst.shape[1] == 1
        for i in range(dst.shape[0]):
            dst[i, 0] = cp.sum(src2 * src1[:, i * w:(i + 1) * w]).astype(cp.float32)

    else:
        dst_h, dst_w = dst.shape
        assert dst_h == dst_w
        w = src2_w // dst_w
        for i in range(dst_w):
            mat = src1[:, i * w:(i + 1) * w]
            dst[i, i] = cp.power(cp.linalg.norm(mat), 2, dtype=cp.float32)

            for j in range(dst_h):
                dst[i, j] = cp.sum(mat * src2[:, j * w:(j + 1) * w], dtype=cp.float32)
                dst[j, i] = dst[i, j]


def update_warping_matrix_ECC(map_matrix, update, motion_type):
    # update = cp.reshape(update, -1)
    if motion_type == MOTION_EUCLIDEAN:
        new_theta = update[0, 0]
        new_theta += cp.arcsin(map_matrix[1, 0])

        map_matrix[0, 2] += update[1, 0]
        map_matrix[1, 2] += update[2, 0]
        cos_new_theta = cp.cos(new_theta)
        map_matrix[0, 0] = cos_new_theta
        map_matrix[1, 1] = cos_new_theta
        sin_new_theta = cp.sin(new_theta)
        map_matrix[1, 0] = sin_new_theta
        map_matrix[0, 1] = -1 * sin_new_theta
    elif motion_type == MOTION_AFFINE:
        map_matrix[0, 0] += update[0, 0]
        map_matrix[1, 0] += update[1, 0]
        map_matrix[0, 1] += update[2, 0]
        map_matrix[1, 1] += update[3, 0]
        map_matrix[0, 2] += update[4, 0]
        map_matrix[1, 2] += update[5, 0]
    elif motion_type == MOTION_TRANSLATION:
        map_matrix[0, 2] += update[0, 0]
        map_matrix[1, 2] += update[1, 0]
    elif motion_type == MOTION_HOMOGRAPHY:
        map_matrix[0, 0] += update[0, 0]
        map_matrix[1, 0] += update[1, 0]
        map_matrix[2, 0] += update[2, 0]
        map_matrix[0, 1] += update[3, 0]
        map_matrix[1, 1] += update[4, 0]
        map_matrix[2, 1] += update[5, 0]
        map_matrix[0, 2] += update[6, 0]
        map_matrix[1, 2] += update[7, 0]
    return


def cp_get_gaussian_kernel(kernel_size, sigma):
    assert kernel_size % 2 == 1
    if sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    dis = cp.arange(kernel_size).reshape(kernel_size, 1)
    dis = cp.abs(dis - kernel_size // 2)
    dis = cp.exp(-1 * dis / (2 * sigma))
    dis = dis / cp.sum(dis)
    return dis


def cp_gaussian_blur(image_float, gauss_filter_size, sigmaX=0, sigmaY=0):
    filt_x = cp_get_gaussian_kernel(gauss_filter_size[0], sigmaX)
    filt_y = cp_get_gaussian_kernel(gauss_filter_size[1], sigmaY)
    filt = cp.matmul(filt_x, filt_y.T)
    convolved_image = convolve(image_float, filt)
    return convolved_image


def cp_findtransformECC(template_image, input_image, warp_matrix=None, motion_type=MOTION_EUCLIDEAN, criteria=None,
                        input_mask=None, gauss_filter_size=5):
    assert isinstance(template_image, (np.ndarray, cp.ndarray))
    assert isinstance(input_image, (np.ndarray, cp.ndarray))

    if motion_type in MOTION_TYPES:
        motion_type = MOTION_TYPES_DICT[motion_type]
    assert motion_type in MOTION_TYPES_CV

    input_image = cp.array(input_image, dtype=cp.float32)

    convert_2_numpy = False
    if warp_matrix is not None:
        assert isinstance(warp_matrix, (np.ndarray, cp.ndarray))
        wm_shape = warp_matrix.shape
        assert wm_shape[1] == 3 and len(wm_shape) == 2
        if motion_type == MOTION_HOMOGRAPHY:
            assert wm_shape[0] == 3
        else:
            assert wm_shape[0] == 2
        if isinstance(warp_matrix, np.ndarray):
            warp_matrix = cp.array(warp_matrix, dtype=cp.float32)
            convert_2_numpy = True
    else:
        warp_matrix = cp.eye(3, 3, dtype=np.float32) if motion_type == MOTION_HOMOGRAPHY else cp.eye(2, 3,
                                                                                                     dtype=cp.float32)

    if criteria is None:
        criteria = CriteriaECC()
    else:
        assert isinstance(criteria, CriteriaECC)

    num_of_pars = ParamTEMP[motion_type]

    hs, ws = template_image.shape[:2]
    # hd, wd = input_image.shape[:2]

    x_grid, y_grid = cp.meshgrid(cp.arange(ws), cp.arange(hs))

    if input_mask is not None:
        assert isinstance(input_mask, np.ndarray)
        pre_mask = (input_mask > 0).astype(np.uint8)
        pre_mask = cp.array(pre_mask)
    else:
        pre_mask = cp.ones_like(input_image).astype(np.uint8)

    template_float = cp.array(template_image, dtype=cp.float32)  ### 不知道是不是要出255
    template_float = cp_gaussian_blur(template_float, (gauss_filter_size, gauss_filter_size), sigmaX=0, sigmaY=0)

    pre_mask_float = cp.copy(pre_mask).astype(cp.float32)
    pre_mask_float = cp_gaussian_blur(pre_mask_float, (gauss_filter_size, gauss_filter_size), 0, 0)
    pre_mask_float = pre_mask_float * (0.5 / 0.95)
    pre_mask = cp.around(pre_mask_float)
    pre_mask_float = pre_mask.astype(cp.float32)

    image_float = input_image
    image_float = cp_gaussian_blur(image_float, (gauss_filter_size, gauss_filter_size), sigmaX=0, sigmaY=0)

    # calculate first order image derivatives
    dx = cp.array([[-0.5, 0, 0.5], [-0.5, 0, 0.5], [-0.5, 0, 0.5]], dtype=cp.float32).reshape(3, 3)

    gradientX = correlate(image_float, dx) * pre_mask_float
    gradientY = correlate(image_float, dx.T) * pre_mask_float

    rho = cp.array(-1)
    last_rho = cp.array(-1 * criteria.eps)

    # convert the array to cp.ndarray & initialize some variables
    warp_matrix = cp.array(warp_matrix)
    jacobian = cp.zeros((hs, num_of_pars * ws), dtype=cp.float32)
    hessian = cp.zeros((num_of_pars, num_of_pars), dtype=cp.float32)
    image_projection = cp.zeros((num_of_pars, 1), dtype=cp.float32)
    template_projection = cp.zeros((num_of_pars, 1), dtype=cp.float32)
    error_projection = cp.zeros((num_of_pars, 1), dtype=cp.float32)

    image_float = cp.array(image_float)
    pre_mask = cp.array(pre_mask)
    template_float = cp.array(template_float)

    ### for 循环部分使用cupy
    for i in range(criteria.num_iter):

        if cp.abs(rho - last_rho) < criteria.eps:
            break

        mat_inv = inv_warp_mat(warp_matrix)
        image_warped = affine_transform(image_float, mat_inv, order=1, mode='opencv')
        gradientX_warped = affine_transform(gradientX, mat_inv, order=1, mode='opencv')
        gradientY_warped = affine_transform(gradientY, mat_inv, order=1, mode='opencv')
        image_mask = affine_transform(pre_mask, mat_inv, order=0, mode='opencv')

        mask_non_zero_num = cp.sum(image_mask > 0).astype(cp.float64)

        image_masked = image_warped * image_mask
        img_mean = cp.sum(image_masked) / mask_non_zero_num
        img_norm_square = cp.sum(cp.power((image_masked - img_mean * image_mask), 2))

        tmp_masked = template_float * image_mask
        tmp_mean = cp.sum(tmp_masked) / mask_non_zero_num

        image_warped = image_masked - img_mean * image_mask
        template_zm = tmp_masked - tmp_mean * image_mask

        if motion_type == MOTION_EUCLIDEAN:  ## euclidean
            image_jacobian_euclidean_ECC(jacobian, gradientX_warped, gradientY_warped, x_grid, y_grid, warp_matrix)
        elif motion_type == MOTION_AFFINE:  ## affine
            image_jacobian_affine_ECC(jacobian, gradientX_warped, gradientY_warped, x_grid, y_grid)
        elif motion_type == MOTION_TRANSLATION:  ## translation
            image_jacobian_translation_ECC(jacobian, gradientX_warped, gradientY_warped)
        elif motion_type == MOTION_HOMOGRAPHY:  ## homography
            image_jacobian_homo_ECC(jacobian, gradientX_warped, gradientY_warped, x_grid, y_grid, warp_matrix)
        else:
            raise NotImplementedError

        project_onto_jacobian_ECC(jacobian, jacobian, hessian)
        hessian_inv = cp.linalg.inv(hessian)

        correlation = cp.sum(template_zm * image_warped).astype(cp.float64)

        # calculate enhanced correlation coefficient (ECC)
        last_rho = rho
        rho = correlation / img_norm_square.astype(cp.float64)

        if cp.isnan(rho):
            raise RuntimeError('NaN encountered')

        project_onto_jacobian_ECC(jacobian, image_warped, image_projection)
        project_onto_jacobian_ECC(jacobian, template_zm, template_projection)

        image_projection_hessian = cp.matmul(hessian_inv, image_projection)
        lambda_n = img_norm_square - cp.sum(image_projection * image_projection_hessian)
        lambda_d = correlation - cp.sum(template_projection * image_projection_hessian)

        if lambda_d < 0:
            raise RuntimeError(
                "The algorithm stopped before its convergence. "
                "The correlation is going to be minimized. "
                "Images may be uncorrelated or non-overlapped")

        lambda_all = lambda_n / lambda_d
        error = lambda_all * template_zm - image_warped
        project_onto_jacobian_ECC(jacobian, error, error_projection)

        delta_p = cp.matmul(hessian_inv, error_projection)

        update_warping_matrix_ECC(warp_matrix, delta_p, motion_type)

    if convert_2_numpy:
        warp_matrix = cp.asnumpy(warp_matrix)
    return rho, warp_matrix
