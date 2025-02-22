from utils import *

#Details of omnidirectional images obtained with the simulator
def main(central,image_path,image_name,specific):
	doc = open('{}.txt'.format(image_path+image_name),'w')
	doc.write('Details of omnidirectional images obtained with the simulator \n')
	doc.write('Image name: {}.png\n'.format(image_name))
	doc.write('Projection model: {}\n'.format(central.cam_type))
	doc.write('Resolution: {}x{} pixels\n'.format(central.width,central.height))
	doc.write('Maximum depth represented: {}\n'.format(central.max_depth))
	if central.cam_type == 'cylindric':
		doc.write('Horizontal x Vertical field of view: {}x{}\n'.format(specific[0],specific[1]))
	elif 'fish_eye' in central.cam_type:
		doc.write('Field Of View: {}\n'.format(specific[1]))
	elif 'catadioptric' in central.cam_type:
		doc.write('Semi latus rectum in meters: {}\n'.format(specific[2]/2.0))
		doc.write('Distance between the camera and the mirror in meters: {}\n'.format(specific[1]))
	elif 'kannala' in central.cam_type:
		doc.write('Field Of View: {}\n'.format(specific[1]))
		doc.write('Kanalla-Brandt\'s calibration parameters: {}\n'.format(specific[2]))
	elif 'scaramuzza' in central.cam_type:
		doc.write('Scaramuzza\'s calibration parameters: {}\n'.format(specific))
	doc.write('Optical center coordinates: {}\n'.format(central.location))
	doc.write('Rotation matrix:{}\n'.format(central.rotation))
	doc.close()

def main_nc(img_name,projection,common,specific,R,mode,loc):
	loc_file = open('cam_loc.txt','r')
	location = loc_file.read().split('\n')	
	rot_file = open('cam_rot.txt','r')
	rotation = rot_file.read().split('\n')
	doc = open('{}.txt'.format(img_name),'w')
	doc.write('Details of omnidirectional images obtained with the simulator \n')
	doc.write('Image name: {}.png\n'.format(img_name))
	doc.write('Projection model: {}\n'.format(projection))
	doc.write('Resolution: {}x{} pixels\n'.format(common[0],common[1]))
	doc.write('View mode: {}\n'.format(mode))
	doc.write('Center coordinates: {}\n'.format(location[loc]))
	doc.write('Maximum depth represented: {}\n'.format(common[3]))
	if 'catadioptric' in projection:
		if specific[0]=='conic':
			doc.write('Aperture angle of the conic mirror: {}\n'.format(specific[2]))
		else:
			doc.write('Radius of the spheric mirror in meters: {}\n'.format(specific[1]))
		doc.write('Distance between the camera and the mirror in meters: {}\n'.format(specific[1]))
	else:
		doc.write('Radius of optical centers: {}\n'.format(specific[0]))
	doc.write('Rotation matrix:\n[{},\t{},\t{};\n{},\t{},\t{};\n{},\t{},\t{}]\n'.format(R[0,0],R[0,1],R[0,2],R[1,0],R[1,1],R[1,2],R[2,0],R[2,1],R[2,2]))
	loc_file.close()
	rot_file.close()
	doc.close()
