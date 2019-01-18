#general imports
import numpy as np
import matplotlib.pyplot as plt 
import random as rand
from astropy import visualization as aviz
import scipy
import os
#lenstronomy imports
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.simulations import Simulation
SimAPI = Simulation()
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Plots.output_plots as lens_plot
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.param_util as param_util
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.image_util as image_util
from lenstronomy.Plots.output_plots import LensModelPlot
from lenstronomy.Workflow.fitting_sequence import FittingSequence
import lenstronomy.Plots.output_plots as out_plot
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
shapeletSet = ShapeletSet()

#********************************************************************************#
#									DES CATALOGUE					 			 #
#********************************************************************************#

cat0=np.loadtxt('gal.npy').reshape(6,111,110) #is needed give the right shape to the file
pix=110 # to have squared images
cat= cat0[:,0:pix,0:pix]

# To Create a mask around the central object
masksize=40 #mask size in pixels
msmin = int(pix/2) - int(masksize/2)
msmax = int(pix/2) + int(masksize/2)
catmask = cat[:,msmin:msmax,msmin:msmax]

# data specifics
sigma_bkg = .1  # background noise per pixel
exp_time = 90  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = pix  # cutout pixel size
deltaPix = 0.27  # pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.958  # full width half max of PSF


#Data
kwargs_data = SimAPI.data_configure(pix, deltaPix, exp_time, sigma_bkg)
data_class = Data(kwargs_data)
#Mask
kwargs_data_mask = SimAPI.data_configure(masksize, deltaPix, exp_time, sigma_bkg)
data_class_mask = Data(kwargs_data_mask)
# PSF specification
kwargs_psf = SimAPI.psf_configure(psf_type='GAUSSIAN', fwhm=fwhm, kernelsize=31, deltaPix=deltaPix, truncate=6, kernel=None)
psf_class = PSF(kwargs_psf)

#********************************************************************************#
#								SOURCE DEFINITION					 			 #
#********************************************************************************#

source_ALL = cat[1]

# CIRCULAR MASK FOR THE SOURCE, where pix is the size of the image, radius is the radius of the mask in pixels

def circular_mask(pix, center, radius):
    Y, X = np.ogrid[:pix, :pix]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

source_cut = circular_mask(pix, (int(pix/2),int(pix/2)),20)*source_ALL #applying a circular mask of 20pix radius around the galaxy

# SHAPELET DECOMPOSITION
x, y = util.make_grid(numPix=numPix, deltapix=deltaPix) # make a coordinate grid
image_1d = util.image2array(source_cut)  # we turn the image in a single 1d array

# we define the shapelet basis set we want the image to decompose in
n_max = 50  # choice of number of shapelet basis functions, 150 is a high resolution number, but takes long
beta = 2  # shapelet scale parameter (in units of resized pixels)

# decompose image and return the shapelet coefficients
coeff_ngc = shapeletSet.decomposition(image_1d, x, y, n_max, beta, 1., center_x=0, center_y=0) 

# reconstruct the galaxy with the shapelet coefficients
image_reconstructed = shapeletSet.function(x, y, coeff_ngc, n_max, beta, center_x=0, center_y=0)
# turn 1d array back into 2d image
image_reconstructed_2d = util.array2image(image_reconstructed) 

#***********************************************************************************************************************************#
#															PLOT 																	#
#***********************************************************************************************************************************#

f, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=False, sharey=False)

ax = axes[0]
im = ax.matshow(source_ALL, origin='lower')
ax.set_title("original image")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

ax = axes[1]
im = ax.matshow(image_reconstructed_2d, origin='lower')
ax.set_title("reconstructed")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

plt.show()

#********************************************************************************#
#								LENS GALAXY LIGHT 					 			 #
#********************************************************************************#

lens_light_model_list = ['SERSIC_ELLIPSE']
lightModel_lens = LightModel(light_model_list=lens_light_model_list)

kwargs_numerics = {'subgrid_res': 2}
i=0
for i in range(len(catmask)):
	data_class_mask.update_data(catmask[i])
	kwargs_data_mask['image_data'] = catmask[i]

	kwargs_model = {'lens_light_model_list': lens_light_model_list}
	kwargs_constraints = {}
	kwargs_numerics_galfit = {'subgrid_res': 2}
	kwargs_likelihood = {'check_bounds': True}

	image_band = [kwargs_data_mask, kwargs_psf, kwargs_numerics_galfit]
	multi_band_list = [image_band]

# lens light model choices as empty arrays so then can be filled is we have more than one profile to be fitted
	fixed_lens_light = []
	kwargs_lens_light_init = []
	kwargs_lens_light_sigma = []
	kwargs_lower_lens_light = []
	kwargs_upper_lens_light = []

# first Sersic component
	fixed_lens_light.append({})
	kwargs_lens_light_init.append({'R_sersic': .1, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0})
	kwargs_lens_light_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
	kwargs_lower_lens_light.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10})
	kwargs_upper_lens_light.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': 10, 'center_y': 10})

	lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
	kwargs_params = {'lens_light_model': lens_light_params}

	fitting_seq = FittingSequence(multi_band_list, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

# n_particles is the number of particles when you can test the parementers, so if you have 5 the code will be tested in 5 positions
# more particles means the fitting is tested in more places so the guess will not converge a local minimum
# but more particles is also time consuming
# n_iterations is the number of iterations in the code and there is no trivial way to know how many are necesary to make the code converge
	fitting_kwargs_list = [{'fitting_routine': 'PSO', 'mpi': False, 'sigma_scale': 1., 'n_particles': 100,'n_iterations': 90}]
	
	lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list)

	lensPlot = LensModelPlot(kwargs_data_mask, kwargs_psf, kwargs_numerics, kwargs_model, lens_result, source_result,
                             lens_light_result, ps_result, arrow_size=0.02, cmap_string="gist_heat")
# this contain the best parameters from fitting the light of the galaxy (light, ellipticity and position)
	LLR = lens_light_result[0]
	

#********************************************************************************#
#									LENS MODEL 						 			 #
#********************************************************************************#

#MAIN DEFLECTOR
# define the lens model of the main deflector
	main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
	kwargs_lens_main = {'theta_E': 1.0, 'e1': LLR['e1'] , 'e2':LLR['e2'],'center_x': LLR['center_x'], 'center_y': LLR['center_y']}
	#kwargs_shear = {'e1': 0.05, 'e2': 0}
	lens_model_list = [main_halo_type]
	kwargs_lens_list = [kwargs_lens_main]


#SUB-HALOS (for now the ammount is = 0 so are not included)
	subhalo_type = 'TNFW'  # We chose spherical NFW profiles, feel free to chose whatever you want.	
# as an example, we render some sub-halos with a very simple distribution to be added on the main lens
	num_subhalo = 0  # number of subhalos to be rendered
# the parameterization of the NFW profiles are:
# - Rs (radius of the scale parameter Rs in units of angles)
# - theta_Rs (radial deflection angle at Rs)
# - center_x, center_y, (position of the centre of the profile in angular units)
	Rs_mean = 0.1
	Rs_sigma = 0.1  # dex scatter
	theta_Rs_mean = 0.05
	theta_Rs_sigma = 0.1 # dex scatter
	r_min, r_max = -2, 2
	Rs_list = 10**(np.log10(Rs_mean) + np.random.normal(loc=0, scale=Rs_sigma, size=num_subhalo))
	theta_Rs_list = 10**(np.log10(theta_Rs_mean) + np.random.normal(loc=0, scale=theta_Rs_sigma, size=num_subhalo))
	center_x_list = np.random.uniform(low=r_min, high=r_max,size=num_subhalo)
	center_y_list = np.random.uniform(low=r_min, high=r_max,size=num_subhalo)
	for i in range(num_subhalo):
	    lens_model_list.append(subhalo_type)
	    kwargs_lens_list.append({'theta_Rs': theta_Rs_list[i], 'Rs': Rs_list[i],'center_x': center_x_list[i], 'center_y': center_y_list[i],'r_trunc': 5*Rs_list[i]})

# now we define a LensModel class of all the lens models combined
	lensModel = LensModel(lens_model_list)

#********************************************************************************#
#								LENSING THE SOURCE					 			 #
#********************************************************************************#

# we define a very high resolution grid for the ray-tracing (needs to be checked to be accurate enough!)
#	numPix = 110  # number of pixels (low res of data)
#	deltaPix = 0.27  # pixel size (low res of data)
	high_res_factor = 5  # higher resolution factor (per axis)
# make the high resolution grid 
	theta_x_high_res, theta_y_high_res = util.make_grid(numPix=numPix*high_res_factor, deltapix=deltaPix/high_res_factor)
# ray-shoot the image plane coordinates (angles) to the source plane (angles)
	beta_x_high_res, beta_y_high_res = lensModel.ray_shooting(theta_x_high_res, theta_y_high_res, kwargs=kwargs_lens_list)

# now we do the same as in Section 2, we just evaluate the shapelet functions in the new coordinate system of the source plane
# Attention, now the units are not pixels but angles! So we have to define the size and position.
# This is simply by chosing a beta (Gaussian width of the Shapelets) and a new center

	source_lensed = shapeletSet.function(beta_x_high_res, beta_y_high_res, coeff_ngc, n_max, beta=.05, center_x=0.1, center_y=0)
# and turn the 1d vector back into a 2d array
	source_lensed_HR = util.array2image(source_lensed)  # map 1d data vector in 2d image

	source_lensed = image_util.re_size(source_lensed_HR, high_res_factor)

	bkg_cut=cat[i][0:20,0:20]
	bkgmed=np.median(bkg_cut)
	sigma=0.985
	source_lensed_conv = scipy.ndimage.filters.gaussian_filter(source_lensed, sigma, mode='nearest', truncate=6)

	exp_time = 90  # exposure time to quantify the Poisson noise level
	background_rms = 0.9# np.sqrt(bkgmed)  # background rms value
	poisson = image_util.add_poisson(source_lensed_conv, exp_time=exp_time)
	bkg = image_util.add_background(source_lensed_conv, sigma_bkd=background_rms)
	noisy_source_lensed =  source_lensed_conv + bkg + poisson

	f, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=False, sharey=False)

	axes[0].imshow(np.log10(source_ALL), cmap='gist_heat', origin="lower")
	axes[0].set_title('Original source')
	axes[1].imshow(np.log10(noisy_source_lensed), cmap='gist_heat', origin="lower")
	axes[1].set_title('Lensed source')
	axes[2].imshow(np.log10(cat[i]), cmap='gist_heat', origin="lower")
	axes[2].set_title('DES Lens galaxy')
	axes[3].imshow(np.log10(noisy_source_lensed+cat[i]),cmap='gist_heat',origin='lower')
	axes[3].set_title('DES Lens galaxy + DES lensed source')
	f.tight_layout()
	plt.show()

