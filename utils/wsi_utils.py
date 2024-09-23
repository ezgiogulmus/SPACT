import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
import openslide
from utils.file_utils import save_pkl
Image.MAX_IMAGE_PIXELS = 933120000


def to_percentiles(scores):
	from scipy.stats import rankdata
	scores = rankdata(scores, 'average')/len(scores) * 100   
	return scores

def screen_coords(scores, coords, top_left, bot_right):
	bot_right = np.array(bot_right)
	top_left = np.array(top_left)
	mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
	scores = scores[mask]
	coords = coords[mask]
	return scores, coords
	
		
class WholeSlideImage(object):
	def __init__(self, path):

		"""
		Args:
			path (str): fullpath to WSI file
		"""

#         self.name = ".".join(path.split("/")[-1].split('.')[:-1])
		self.name = os.path.splitext(os.path.basename(path))[0]
		self.full_path = path
		self.wsi = openslide.open_slide(path)
		self.level_downsamples = self._assertLevelDownsamples()
		self.level_dim = self.wsi.level_dimensions
	
		self.contours_tissue = None
		self.contours_tumor = None
		self.hdf5_file = None

	def saveSegmentation(self, mask_file):
		# save segmentation results using pickle
		asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
		save_pkl(mask_file, asset_dict)

	def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
							filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[], target_downsample=None):
		"""
			Segment the tissue via HSV -> Median thresholding -> Binary threshold
		"""
		
		def _filter_contours(contours, hierarchy, filter_params):
			"""
				Filter contours by: area.
			"""
			filtered = []

			# find indices of foreground contours (parent == -1)
			hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
			all_holes = []
			
			# loop through foreground contour indices
			for cont_idx in hierarchy_1:
				# actual contour
				cont = contours[cont_idx]
				# indices of holes contained in this contour (children of parent contour)
				holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
				# take contour area (includes holes)
				a = cv2.contourArea(cont)
				# calculate the contour area of each hole
				hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
				# actual area of foreground contour region
				a = a - np.array(hole_areas).sum()
				if a == 0: continue
				if tuple((filter_params['a_t'],)) < tuple((a,)): 
					filtered.append(cont_idx)
					all_holes.append(holes)


			foreground_contours = [contours[cont_idx] for cont_idx in filtered]
			
			hole_contours = []

			for hole_ids in all_holes:
				unfiltered_holes = [contours[idx] for idx in hole_ids ]
				unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
				# take max_n_holes largest holes by area
				unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
				filtered_holes = []
				
				# filter these holes
				for hole in unfilered_holes:
					if cv2.contourArea(hole) > filter_params['a_h']:
						filtered_holes.append(hole)

				hole_contours.append(filtered_holes)

			return foreground_contours, hole_contours
		
		if target_downsample is None:
			img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
		else:
			img = self.selected_page.asarray()
			
		img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
		img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
		
	   
		# Thresholding
		if use_otsu:
			_, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
		else:
			_, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

		# Morphological closing
		if close > 0:
			kernel = np.ones((close, close), np.uint8)
			img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

		scale = self.level_downsamples[seg_level] if target_downsample is None else (target_downsample, target_downsample)
		scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
		filter_params = filter_params.copy()
		filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
		filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
		
		# Find and filter contours
		contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
		hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
		if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

		self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
		self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

		if len(keep_ids) > 0:
			contour_ids = set(keep_ids) - set(exclude_ids)
		else:
			contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

		self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
		self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]
		

	def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), 
					line_thickness=250, max_size=None, top_left=None, bot_right=None, target_downsample=None, view_slide_only=False,
					number_contours=False, seg_display=True, annot_display=True):
		
		downsample = self.level_downsamples[vis_level] if target_downsample is None else (target_downsample, target_downsample)
		scale = [1/downsample[0], 1/downsample[1]]
		
		if top_left is not None and bot_right is not None:
			top_left = tuple(top_left)
			bot_right = tuple(bot_right)
			w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
			region_size = (w, h)
		else:
			top_left = (0,0)
			if target_downsample is None:
				region_size = self.level_dim[vis_level]
			else:
				region_size = self.selected_page.shape[:2]

		if target_downsample is None:
			img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
		else:
			img = self.selected_page.asarray()

		if not view_slide_only:
			offset = tuple(-(np.array(top_left) * scale).astype(int))
			line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
			if self.contours_tissue is not None and seg_display:
				if not number_contours:
					cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), 
									 -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

				else: # add numbering to each contour
					for idx, cont in enumerate(self.contours_tissue):
						contour = np.array(self.scaleContourDim(cont, scale))
						M = cv2.moments(contour)
						cX = int(M["m10"] / (M["m00"] + 1e-9))
						cY = int(M["m01"] / (M["m00"] + 1e-9))
						# draw the contour and put text next to center
						cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
						cv2.putText(img, "{}".format(idx), (cX, cY),
								cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

				for holes in self.holes_tissue:
					cv2.drawContours(img, self.scaleContourDim(holes, scale), 
									 -1, hole_color, line_thickness, lineType=cv2.LINE_8)
			
			if self.contours_tumor is not None and annot_display:
				cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), 
								 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)
		
		img = Image.fromarray(img)
	
		w, h = img.size
		if target_downsample is not None:
			img = img.resize((int(w/target_downsample), int(h/target_downsample)))

		if max_size is not None and (w > max_size or h > max_size):
			resizeFactor = max_size/w if w > h else max_size/h
			img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
	   
		return img
	
	@staticmethod
	def isInHoles(holes, pt, patch_size):
		for hole in holes:
			if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
				return 1
		
		return 0

	@staticmethod
	def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
		if cont_check_fn(pt):
			if holes is not None:
				return not WholeSlideImage.isInHoles(holes, pt, patch_size)
			else:
				return 1
		return 0
	
	@staticmethod
	def scaleContourDim(contours, scale):
		return [np.array(cont * scale, dtype='int32') for cont in contours]

	@staticmethod
	def scaleHolesDim(contours, scale):
		return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

	def _assertLevelDownsamples(self):
		level_downsamples = []
		dim_0 = self.wsi.level_dimensions[0]
		
		for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
			estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
			level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
		
		return level_downsamples

	def visHeatmap(self, scores, coords, vis_level=-1, 
				   top_left=None, bot_right=None,
				   patch_size=(256, 256), 
				   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4, 
				   blur=False, overlap=0.0, 
				   segment=True, use_holes=True,
				   convert_to_percentiles=False, 
				   binarize=False, thresh=0.5,
				   max_size=None,
				   custom_downsample = 1,
				   cmap='coolwarm'):

		"""
		Args:
			scores (numpy array of float): Attention scores 
			coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
			vis_level (int): WSI pyramid level to visualize
			patch_size (tuple of int): Patch dimensions (relative to lvl 0)
			blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
			canvas_color (tuple of uint8): Canvas color
			alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
			blur (bool): apply gaussian blurring
			overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
			segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
							self.contours_tissue and self.holes_tissue are not None
			use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
			convert_to_percentiles (bool): whether to convert attention scores to percentiles
			binarize (bool): only display patches > threshold
			threshold (float): binarization threshold
			max_size (int): Maximum canvas size (clip if goes over)
			custom_downsample (int): additionally downscale the heatmap by specified factor
			cmap (str): name of matplotlib colormap to use
		"""

		if vis_level < 0:
			vis_level = self.wsi.get_best_level_for_downsample(32)

		downsample = self.level_downsamples[vis_level]
		scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
				
		if len(scores.shape) == 2:
			scores = scores.flatten()

		if binarize:
			if thresh < 0:
				threshold = 1.0/len(scores)
				
			else:
				threshold =  thresh
		
		else:
			threshold = 0.0

		##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
		if top_left is not None and bot_right is not None:
			scores, coords = screen_coords(scores, coords, top_left, bot_right)
			coords = coords - top_left
			top_left = tuple(top_left)
			bot_right = tuple(bot_right)
			w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
			region_size = (w, h)

		else:
			region_size = self.level_dim[vis_level]
			top_left = (0,0)
			bot_right = self.level_dim[0]
			w, h = region_size

		patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
		coords = np.ceil(coords * np.array(scale)).astype(int)
		
		print('\ncreating heatmap for: ')
		print('top_left: ', top_left, 'bot_right: ', bot_right)
		print('w: {}, h: {}'.format(w, h))
		print('scaled patch size: ', patch_size)
		print(scores.shape, coords.shape)
		###### normalize filtered scores ######
		if convert_to_percentiles:
			scores = to_percentiles(scores)
			scores /= 100
		
		######## calculate the heatmap of raw attention scores (before colormap) 
		# by accumulating scores over overlapped regions ######
		
		# heatmap overlay: tracks attention score over each pixel of heatmap
		# overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
		overlay = np.full(np.flip(region_size), 0).astype(float)
		counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
		count = 0
		for idx in range(len(coords)):
			score = scores[idx]
			coord = coords[idx]
			if score >= threshold:
				if binarize:
					score=1.0
					count+=1
			else:
				score=0.0
			# accumulate attention
			overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
			# accumulate counter
			counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

		if binarize:
			print('\nbinarized tiles based on cutoff of {}'.format(threshold))
			print('identified {}/{} patches as positive'.format(count, len(coords)))
		
		# fetch attended region and average accumulated attention
		zero_mask = counter == 0

		if binarize:
			overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
		else:
			overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
		del counter 
		if blur:
			overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

		if segment:
			tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
			# return Image.fromarray(tissue_mask) # tissue mask
		
		if not blank_canvas:
			# downsample original image and use as canvas
			img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
		else:
			# use blank canvas
			img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

		#return Image.fromarray(img) #raw image

		print('\ncomputing heatmap image')
		print('total of {} patches'.format(len(coords)))
		twenty_percent_chunk = max(1, int(len(coords) * 0.2))

		if isinstance(cmap, str):
			cmap = plt.get_cmap(cmap)
		
		for idx in range(len(coords)):
			if (idx + 1) % twenty_percent_chunk == 0:
				print('progress: {}/{}'.format(idx, len(coords)))
			
			score = scores[idx]
			coord = coords[idx]
			if score >= threshold:

				# attention block
				raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
				
				# image block (either blank canvas or orig image)
				img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

				# color block (cmap applied to attention block)
				color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

				if segment:
					# tissue mask block
					mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
					# copy over only tissue masked portion of color block
					img_block[mask_block] = color_block[mask_block]
				else:
					# copy over entire color block
					img_block = color_block

				# rewrite image block
				img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()
		
		#return Image.fromarray(img) #overlay
		print('Done')
		del overlay

		if blur:
			img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

		if alpha < 1.0:
			img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)
		
		img = Image.fromarray(img)
		w, h = img.size

		if custom_downsample > 1:
			img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

		if max_size is not None and (w > max_size or h > max_size):
			resizeFactor = max_size/w if w > h else max_size/h
			img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
	   
		return img

	
	def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
		print('\ncomputing blend')
		downsample = self.level_downsamples[vis_level]
		w = img.shape[1]
		h = img.shape[0]
		block_size_x = min(block_size, w)
		block_size_y = min(block_size, h)
		print('using block size: {} x {}'.format(block_size_x, block_size_y))

		shift = top_left # amount shifted w.r.t. (0,0)
		for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
			for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
				#print(x_start, y_start)

				# 1. convert wsi coordinates to image coordinates via shift and scale
				x_start_img = int((x_start - shift[0]) / int(downsample[0]))
				y_start_img = int((y_start - shift[1]) / int(downsample[1]))
				
				# 2. compute end points of blend tile, careful not to go over the edge of the image
				y_end_img = min(h, y_start_img+block_size_y)
				x_end_img = min(w, x_start_img+block_size_x)

				if y_end_img == y_start_img or x_end_img == x_start_img:
					continue
				#print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))
				
				# 3. fetch blend block and size
				blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img] 
				blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)
				
				if not blank_canvas:
					# 4. read actual wsi block as canvas block
					pt = (x_start, y_start)
					canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))     
				else:
					# 4. OR create blank canvas block
					canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

				# 5. blend color block and canvas block
				img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
		return img

	def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
		print('\ncomputing foreground tissue mask')
		tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
		contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
		offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

		contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
		contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
		for idx in range(len(contours_tissue)):
			cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)

			if use_holes:
				cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
			# contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)
				
		tissue_mask = tissue_mask.astype(bool)
		print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
		return tissue_mask