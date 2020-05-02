import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from utils import *


class ToTensor(object):
    def __call__(self, img):
        return torch.FloatTensor(img)
    

def get_absmax(dataset):
    absmax = 0.
    for img in tqdm(dataset):
        if dataset.target is not None:
            img = torch.FloatTensor(img[0])
        else:
            img = torch.FloatTensor(img)
        absmax = max(absmax, img.abs().max().item())
        del img
    return absmax

class AbsMaxScale(object):
    def __init__(self, absmax):
        self.absmax = absmax
        
    def __call__(self, img):
        return img / self.absmax


class Pad(object):
    """
    Pads image with predefined padding size 
    Applicable: sMRI, fMRI-slice, fMRI
    """
    def __init__(self, padding=(0, 0, 0), value=0):
        self.padding = padding
        self.value = np.float64(value)
    
    def __call__(self, img):
        if self.padding == (0, 0, 0):
                return img
        
        img_shape = img.shape[1:]
        if len(img_shape) == 4: # fmri
            padded_shape = np.array(img_shape)
            padded_shape[:3] += np.array(self.padding) * 2
            padded_img = np.full(padded_shape, self.value)
            padded_img[self.padding[0]:self.padding[0] + img_shape[0], 
                       self.padding[1]:self.padding[1] + img_shape[1], 
                       self.padding[2]:self.padding[2] + img_shape[2], :] = img[0]
        else:
            padded_shape = np.array(img_shape) + np.array(self.padding) * 2
            padded_img = np.full(padded_shape, self.value)
            padded_img[self.padding[0]:self.padding[0] + img_shape[0], 
                       self.padding[1]:self.padding[1] + img_shape[1], 
                       self.padding[2]:self.padding[2] + img_shape[2]] = img[0]
        return padded_img[np.newaxis, :]
    
    
def create_centered_identity_transformation_field(shape, spacings):
    """
    Create 2D or 3D centered identity transformation field.
    """
    coords = []
    for i, size in enumerate(shape):
        spacing = spacings[i]
        coords.append(torch.linspace(
        -(size - 1) / 2 * spacing,
        (size - 1) / 2 * spacing,
        size))
    permutation = np.roll(np.arange(len(coords) + 1), -1)
    return torch.transpose(torch.meshgrid(*coords, indexing="ij"), permutation)

def create_control_grid_for_cubic_interp(transformed_image_shape,
                                       transformed_image_spacings_um,
                                         control_grid_spacings_pix):
    """
    Create a control grid with optimal size for cubic interpolation.
    """
    grid_shape = np.zeros(len(transformed_image_shape), dtype=int)
    for comp in range(len(transformed_image_shape)):
        spacing_pix = float(control_grid_spacings_pix[comp])
        num_elem = float(transformed_image_shape[comp])
        if num_elem % 2 == 0:
            grid_shape[comp] = np.ceil((num_elem - 1) / (2 * spacing_pix) +
                                 0.5) * 2 + 2
        else:
            grid_shape[comp] = np.ceil((num_elem - 1) / (2 * spacing_pix)) * 2 + 3
    control_grid_spacings_um = torch.mul(
      torch.tensor(control_grid_spacings_pix, dtype=tf.float32),
      transformed_image_spacings_um)
#     maybe not torch.mul check!!!
    control_grid = create_centered_identity_transformation_field(
      grid_shape, control_grid_spacings_um)
    control_grid.set_shape(np.append(grid_shape, len(control_grid_spacings_pix)))
    return control_grid

def rotation_matrix(rot):
    """
    Creates a 3D rotation matrix.
  
    """
#     sin_angle = np.sin(rot)
#     cos_angle = np.cos(rot)
        
    rotation_x = np.array([[1.0, 0.0, 0.0],[0.0, np.cos(rot[0]), -np.sin(rot[0])],[0.0,np.sin(rot[0]),np.cos(rot[0])]])
    rotation_y = np.array([[np.cos(rot[1]),0.0,np.sin(rot[1])],[0.0, 1.0, 0.0],[-np.sin(rot[1]),0.0,np.cos(rot[1])]])
    rotation_z = np.array([[np.cos(rot[2]),-np.sin(rot[2]),0.0],[np.sin(rot[2]),np.cos(rot[2]),0.0], [0.0, 0.0,1.0]])
    rotation_mat = torch.from_numpy(rotation_x @ rotation_y @ rotation_z).float()
    return rotation_mat

def shearing_matrix(shearing_coefs):
    """
      Creates a 3D shearing matrix.
    """
#     shearing = np.array([[1., shearing_coefs[0], shearing_coefs[1]],
#               [shearing_coefs[2], 1., shearing_coefs[3]],
#               [shearing_coefs[4], shearing_coefs[5], 1.]])
    shearing = np.array([[1., 0.1, 0.1],
               [0.1, 1., 0.1],
               [0.1, 0.1, 1.]])
    return torch.from_numpy(shearing).float()

def deformation_field(
    raw_image_center_pos_pix, raw_image_element_size_um,
    net_input_spatial_shape, net_input_element_size_um,
    control_grid_spacings_pix, deformations_magnitudes_um, rotation_angles,
    scale_factors, mirror_factors, shearing_coefs, cropping_offset_pix):
  """Create a 3D deformation field.
  Creates a dense 3D deformation field for affine and elastic deformations.
  """
  # Set up the centered control grid for identity transform in real world
  # coordinates.
  control_grid = create_control_grid_for_cubic_interp(
      net_input_spatial_shape, net_input_element_size_um,
      control_grid_spacings_pix)

  # Add random deformation.
  control_grid += deformations_magnitudes_um * tf.random.normal(
      shape=control_grid.shape)

  # Apply affine transformation and transform units to raw image pixels.
  scale_to_pix = 1. / raw_image_element_size_um
  affine = tf.matmul(
      create_3x3_rotation_matrix(rotation_angles),
      tf.diag(scale_factors * mirror_factors * scale_to_pix))
  affine_shearing = tf.matmul(
      affine, create_3x3_shearing_matrix(shearing_coefs))

  control_grid = tf.reshape(
      tf.matmul(tf.reshape(control_grid, [-1, 3]), affine_shearing),
      control_grid.shape)

  # Translate to cropping position.
  control_grid += raw_image_center_pos_pix + cropping_offset_pix

  # Create the dense deformation field for the image.
  dense_deformation_field = augmentation_ops.cubic_interpolation3d(
      control_grid, control_grid_spacings_pix, net_input_spatial_shape)

  return dense_deformation_field
                               
class BrightnessContrast(object):
    
    def __init__(self, probability=0.5, alpha = None, beta = None):
        self.probability = probability
        self.alpha = round(np.random.uniform(1.0,3.0),3) # contrast control
        self.beta = np.random.uniform(1,100)    # brightness control

    
    def __call__(self, img):
        """
        for future mb https://medium.com/@fanzongshaoxing/adjust-local-brightness-for-image-augmentation-8111c001059b

        """
#         print("shape")
#         print(img.shape)
        if not np.random.uniform(0, 1.1) < self.probability:
             return img
        img = torch.squeeze(torch.from_numpy(img).float())
        augmented_img = self.alpha*img + self.beta
        return torch.unsqueeze(augmented_img,0)
                
class Gaussian_blur(object):
    
    def __init__(self, probability=0.5, size = 4.0, mean = 0.0, var = None):
        self.probability = probability
        self.size = size
        self.mean = mean
        self.var = round(np.random.uniform(0.1,0.9),1)    
        self.sigma = self.var**0.5
    
    def __call__(self, img):
        """

        """
         #if not np.random.uniform(0, 1.1) < self.probability:
             #return img
        img = torch.squeeze(torch.from_numpy(img).float())
        h,w,d = img.shape
        gauss = np.random.normal(self.mean,self.sigma,(h,w,d))
#         gauss_kernel = torch.from_numpy(gauss.reshape(self.size,self.size,self.size)).float()
        # Gaussian kernel
        gauss_kernel = [[1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256],
                            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                            [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
                            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                            [1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256]]
    
        gaus_x = np.zeros((h, w - self.size + 1, d), dtype='float64')
        for i, v, l in enumerate(gauss_kernel):
            gaus_x += v * img[:, i : w - self.size + i + 1,:]
        gaus_y = np.zeros((gaus_x.shape[0] - self.size + 1, gaus_x.shape[1],gaus_x.shape[2] ))
        for i, v, l in enumerate(gauss_kernel):
            gaus_y += v * gaus_x[i : img.shape[0]  - self.size + i + 1]
        augmented_img = np.zeros((gaus_y.shape[0],gaus_x.shape[1], gaus_x.shape[2] - self.size + 1))
        for i, v, l in enumerate(gauss_kernel):
            augmented_img += v * gaus_z[:,:,i : img.shape[3]  - self.size + i + 1]
        
        

#         # Middle of the kernel
#         offset = len(kernel) // 2
#         # Compute convolution between intensity and kernels
#         for x in range(offset, input_image.width - offset):
#             for y in range(offset, input_image.height - offset):
#                 acc = [0, 0, 0]
#                 for a in range(len(kernel)):
#                     for b in range(len(kernel)):
#                         xn = x + a - offset
#                         yn = y + b - offset
#                         pixel = input_pixels[xn, yn]
#                         acc[0] += pixel[0] * kernel[a][b]
#                         acc[1] += pixel[1] * kernel[a][b]
#                         acc[2] += pixel[2] * kernel[a][b]


        return torch.unsqueeze(augmented_img,0)
                         
class GaussNoise(object):
    
    def __init__(self, probability=0.5, mean = 0.0, var = None):
        self.probability = probability
        self.mean = mean
        self.var = round(np.random.uniform(0.1,0.9),1)    
        self.sigma = self.var**0.5
    
    def __call__(self, img):
        """

        """
        if not np.random.uniform(0, 1.1) < self.probability:
            return img
        img = torch.squeeze(torch.from_numpy(img).float())
        h,w,d = img.shape
        gauss = np.random.normal(self.mean,self.sigma,(h,w,d))
        gauss = torch.from_numpy(gauss.reshape(h,w,d)).float()
        augmented_img = img + gauss
#         print(self.var)
#         print(self.sigma)
        return torch.unsqueeze(augmented_img,0)
    
           
class Shear(object):
    
    def __init__(self, probability=0.5, min_shear = 0.1, max_shear = 0.35):
        self.probability = probability 
        self.min_shear = min_shear
        self.max_shear = max_shear
    
    def __call__(self, img):
        """
       
        """
        if not np.random.uniform(0, 1.1) < self.probability:
            return img
        img = torch.squeeze(torch.from_numpy(img).float())
        h,w,d = img.shape
        augmented_img = torch.zeros([h, w, d])
        coef = np.round(np.random.uniform(self.min_shear, self.max_shear, 6),2)
        shear_mat = shearing_matrix(coef)
        for i in tqdm(range(0,w)):
            for j in range(0,h):
                for l in range(0,d):
                    new_coord = shear_mat @ torch.tensor([j,i,l]).float().unsqueeze(1)
                    if (torch.sum(torch.ge(new_coord, 0)) == 3) & (torch.lt(new_coord[0],h)) & (torch.lt(new_coord[1],w)) & (torch.lt(new_coord[2],d)):
                        augmented_img[j,i,l] = img[new_coord[0].long(), new_coord[1].long(), new_coord[2].long()]
        return torch.unsqueeze(augmented_img,0)
    
class HorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        """
        Args:
        
        """
        
        if not np.random.uniform(0, 1.1) < self.probability:
             return img
        img = torch.squeeze(torch.from_numpy(img).float())
        h,w,d = img.shape
        augmented_img = torch.zeros([h, w, d])
        for i in tqdm(range(0,w)):
            xi = w - i - 1
            for j in range(0,h):
                for l in range(0,d):
                    new_coord = torch.tensor([j,xi,l]).float().unsqueeze(1)
                    if (torch.sum(torch.ge(new_coord, 0)) == 3) & (torch.le(new_coord[0],h)) & (torch.le(new_coord[1],w)) & (torch.le(new_coord[2],d)):
                        augmented_img[j,i,l] = img[new_coord[0].long(), new_coord[1].long(), new_coord[2].long()]
        return torch.unsqueeze(augmented_img,0)
                
class Rotate(object):
    
    def __init__(self, probability=0.5, min_rotation = 15, max_rotation=15):
        self.probability = probability
        self.min_rotation = -abs(min_rotation)   
        self.max_rotation = abs(max_rotation)  

    
    def __call__(self, img):
        """
        Args:
            img: Image to be rotated.

        Returns:
         Rotated image.
        """
        
#         if not np.random.uniform(0, 1.1) < self.probability:
#             return img
        img = torch.squeeze(torch.from_numpy(img).float())
        h,w,d = img.shape
        print(h,w,d)
        augmented_img = torch.zeros([h, w, d])
        angle = np.random.randint(self.min_rotation, self.max_rotation, 3)
        rotation_mat = rotation_matrix(np.radians(angle))
        for i in tqdm(range(0,w)):
            for j in range(0,h):
                for l in range(0,d):
                    new_coord = rotation_mat @ torch.tensor([j,i,l]).float().unsqueeze(1)
                    print(i,j,l)
                    if (torch.sum(torch.ge(new_coord, 0)) == 3) & (torch.le(new_coord[0],h)) & (torch.le(new_coord[1],w)) & (torch.le(new_coord[2],d)):
                        augmented_img[j,i,l] = img[new_coord[0].long(), new_coord[1].long(), new_coord[2].long()]
        return torch.unsqueeze(augmented_img,0)