# Faster OBJ, by Christopher Night
# This script is based on (and intended to replace) the one found here:
# http://www.pygame.org/wiki/OBJFileLoader?parent=CookBook
# The OBJ objects produced with this script should be pickleable,
# and they should load from the pickle file quickly, and render quickly.

import pygame, ctypes
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from numpy import array
import numpy as np
import cv2
import render.face_pose_class as face_pose_class

# Convert a list to a ctype string for pickling
def toctype(val, btype = ctypes.c_float):
    a = (btype * len(val))(*val)
    return ctypes.string_at(a, ctypes.sizeof(a))

# Convert an unpickled string to a ctype array
def fromctype(s, btype = ctypes.c_float):
    return (btype * (len(s) // ctypes.sizeof(btype))).from_buffer_copy(s)

# TODO: cache textures between files
class MTL(object):
	"""Material info for a 3-D model as represented in an MTL file"""
	def __init__(self, filename):
		"""Read data from the file"""
		self.contents = {}
		mtl = None
		for line in open(filename, "r"):
			if line.startswith('#'): continue
			values = line.split()
			if not values: continue
			if values[0] == 'newmtl':
				mtl = self.contents[values[1]] = {}
			elif mtl is None:
				raise (ValueError, "mtl file doesn't start with newmtl stmt")
			elif values[0] == 'map_Kd':
				mtl[values[0]] = values[1]
				surf = pygame.image.load(mtl['map_Kd'])
				mtl["image"] = pygame.image.tostring(surf, 'RGBA', 1)
				mtl["ix"], mtl["iy"] = surf.get_rect().size
			else:
				mtl[values[0]] = map(float, values[1:])

	def generate(self):
		"""Generate the textures necessary for any materials"""
		for mtl in self.contents.values():
			if "map_Kd" not in mtl: continue
			texid = mtl['texture_Kd'] = glGenTextures(1)
			glBindTexture(GL_TEXTURE_2D, texid)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mtl["ix"], mtl["iy"], 0, GL_RGBA, GL_UNSIGNED_BYTE, mtl["image"])

	def bind(self, material):
		mtl = self.contents[material]
		if 'texture_Kd' in mtl:
			glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
		else:
			glBindTexture(GL_TEXTURE_2D, 0)
			glColor(*mtl['Kd'])


class OBJ(object):
	"""Geometric info for a 3-D model as represented in an OBJ file.
	Base class uses fixed function."""
	swapyz = True
	reorder_materials = True  # Allow optimization by reordering materials
	reorder_polygons = True  # Allow optimization by reordering materials
	treat_polygon = False  # Set to False to treat triangles and quads specially
	use_list = True  # Use a display list
	use_ctypes = True
	generate_on_init = False  # Generate display list when loaded from OBJ
	generate_on_load = True  # Generate display list when loaded from pickle file

	def __init__(self, filename):
		self.parse(filename)
		self.process()
		if self.generate_on_init:
			self.generate()
		else:
			self.generated = False

	def getpoint(self, values):
		"""Extract a 3-d point"""
		indices = (0,2,1) if self.swapyz else (0,1,2)
		return tuple(float(values[i]) for i in indices)

	@staticmethod
	def getentry(entrylist, key, join = True):
		"""Get the value corresponding to a given key in a list of key,value pairs.
		The keys in this list are not necessarily unique. If join is False, a value may
		only be returned if it's last in the list. If necessary, a new key,value pair
		is appended to the list, and the new value is returned."""
		if not entrylist:
			r = None
		elif key == entrylist[-1][0]:
			r = entrylist[-1][1]
		elif not join:
			r = None
		else:
			d = dict(entrylist)
			r = d[key] if key in d else None
		if r is None:
			r = []
			entrylist.append((key, r))
		return r

	def parse(self, filename):
		"""Read data from the file"""
		self.vertices = []
		self.normals = []
		self.texcoords = []
		self.mfaces = []

		material = None
		lines = open(filename, "r").read().replace("\\\n", " ").splitlines()
		for line in lines:
			values = line.partition('#')[0].split()
			if not values: continue
			elif values[0] == 's':
				smoothing = values[1] not in ("off", "0")
				# TODO: smoothing
			elif values[0] == 'v':
				self.vertices.append(self.getpoint(values[1:]))
			elif values[0] == 'vn':
				self.normals.append(self.getpoint(values[1:]))
			elif values[0] == 'vt':
				if len(values) != 3:
					raise NotImplementedError("Only 2-d textures implemented.")
				# self.texcoords.append(map(float, values[1:3]))
				self.texcoords.append(values[1:3])

			elif values[0] in ('usemtl', 'usemat'):
				if len(values) < 2:
					raise NotImplementedError # use white material
				else:
					material = values[1]
			elif values[0] == 'mtllib':
				self.mtl = MTL(values[1])  # TODO: multiple files?
			elif values[0] == 'p':
				raise NotImplementedError
			elif values[0] == 'l':
				raise NotImplementedError
			elif values[0] == 'f':
				face = []
				texcoords = []
				norms = []
				for vs in values[1:]:
					w = vs.split('/')
					v = int(w[0])
					vt = int(w[1]) if len(w) > 1 and w[1] else 0
					vn = int(w[2]) if len(w) > 2 and w[2] else 0
					face.append(v if v >= 0 else len(self.vertices) + 1 + v)
					texcoords.append(vt if vt >= 0 else len(self.texcoords) + 1 + vt)
					norms.append(vn if vn >= 0 else len(self.normals) + 1 + vn)

				tfaces = self.getentry(self.mfaces, material, self.reorder_materials)
				nvs = 5 if self.treat_polygon else min(len(face), 5)
				key = nvs, bool(texcoords[0]), bool(norms[0])
				value = face, norms, texcoords
				if nvs == 5:
					tfaces.append((key, value))
				else:
					faces = self.getentry(tfaces, key, self.reorder_polygons)
					if faces:
						faces[0].extend(face)
						faces[1].extend(norms)
						faces[2].extend(texcoords)
					else:
						faces.extend(value)
			elif values[0] in ('vp', 'cstype', 'deg', 'bmat', 'step', 'curv', 'curv2', 'surf',
			    'parm', 'trim', 'hole', 'scrv', 'sp', 'end', 'con', 'g', 'ng', 'o',
			    'bevel', 'c_interp', 'd_interp', 'lod', 'maplib', 'usemap', 'shadow_obj',
			    'trace_obj', 'ctech', 'stech'):
				raise NotImplementedError("Non-impemented OBJ element %s" % values[0])
			elif values[0] in ('fo', 'bsp', 'bzp', 'cdc', 'cdp', 'res'):
				raise NotImplementedError("Deprecated OBJ element %s" % values[0])
			else:
				raise NotImplementedError("Unrecognized OBJ element %s" % values[0])

	def process(self):
		"""Processing that must be done after parsing before generating"""
		pass

	def generate(self):
		"""Build the display list, if it's used"""
		self.mtl.generate()
		if self.use_list:
			self.gl_list = glGenLists(1)
			glNewList(self.gl_list, GL_COMPILE)
			self.base_render()
			glEndList()
		self.generated = True

	def base_render(self):
		glEnable(GL_TEXTURE_2D)
		glFrontFace(GL_CCW)
		self.basic_render()
		glDisable(GL_TEXTURE_2D)

	def basic_render(self):
		for material, tfaces in self.mfaces:
			self.mtl.bind(material)
			for (nvs, dotex, donorm), (vertices, normals, texture_coords) in tfaces:
				shape = [GL_TRIANGLES, GL_QUADS, GL_POLYGON][nvs-3]
				glBegin(shape)
				for i in range(len(vertices)):
					if donorm: glNormal3fv(self.normals[normals[i] - 1])
					if dotex: glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
					glVertex3fv(self.vertices[vertices[i] - 1])
				glEnd()

	def render(self):
		if not self.generated:
			self.generate()
		if self.use_list:
			glCallList(self.gl_list)
		else:
			self.base_render()

	def __setstate__(self, state):
		self.__dict__ = state
		if self.generate_on_load:
			self.generate()
		else:
			self.generated = False

	def __del__(self):
		if self.generated and self.use_list and glDeleteLists:
			glDeleteLists(self.gl_list,1)

class OBJ_array(OBJ):
	"""3-D model using vertex arrays"""
	
	def process(self):
		"""Build index list for vertex arrays"""
		self.indices = []
		self.ivertices = []
		self.inormals = []
		self.itexcoords = []
		imap = {}
		for material, tfaces in self.mfaces:
			mindices = []
			for (nvs, dotex, donorm), (vs, vns, vts) in tfaces:
				inds = []
				for v, vn, vt in zip(vs, vns, vts):
					ivertex = tuple(self.vertices[v-1])
					inorm = tuple(self.normals[vn-1]) if vn else (0.,0.,0.)
					itcoord = tuple(self.texcoords[vt-1]) if vt else (0.,0.)
					key = ivertex, inorm, itcoord
					if key not in imap:
						imap[key] = len(imap)
						self.ivertices.extend(ivertex)
						self.inormals.extend(inorm)
						self.itexcoords.extend(itcoord)
					inds.append(imap[key])
				mindices.append((nvs, dotex, donorm, inds))
			self.indices.append((material, mindices))
		if self.use_ctypes:
			self.ivertices = toctype(self.ivertices)
			self.inormals = toctype(self.inormals)
			self.itexcoords = toctype(self.itexcoords)
		del self.vertices, self.normals, self.texcoords

	def basic_render(self):
		glVertexPointer(3, GL_FLOAT, 0, self.ivertices)
		glNormalPointer(GL_FLOAT, 0, self.inormals)
		glTexCoordPointer(2, GL_FLOAT, 0, self.itexcoords)

		glEnableClientState(GL_VERTEX_ARRAY)
		texon, normon = None, None
		for material, mindices in self.indices:
			self.mtl.bind(material)
			for nvs, dotex, donorm, ilist in mindices:
				if donorm != normon:
					normon = donorm
					(glEnableClientState if donorm else glDisableClientState)(GL_NORMAL_ARRAY)
				if dotex != texon:
					texon = dotex
					(glEnableClientState if dotex else glDisableClientState)(GL_TEXTURE_COORD_ARRAY)
				shape = [GL_TRIANGLES, GL_QUADS, GL_POLYGON][nvs-3]
				glDrawElements(shape, len(ilist), GL_UNSIGNED_INT, ilist)

class OBJ_vbo(OBJ):
	"""3-D model using vertex buffer objects"""

	def __init__(self, filename):
		self.vbo_v = None
		OBJ.__init__(self, filename)

	def process(self):
		"""Build index list for VBOs"""
		self.indices = []
		vertices = []
		normals = []
		texcoords = []
		for material, tfaces in self.mfaces:
			mindices = []
			for (nvs, dotex, donorm), (vs, vns, vts) in tfaces:
				j = len(vertices)
				for v, vn, vt in zip(vs, vns, vts):
					ivertex = tuple(self.vertices[v-1])
					inorm = tuple(self.normals[vn-1]) if vn else (0.,0.,0.)
					itcoord = tuple(self.texcoords[vt-1]) if vt else (0.,0.)
					vertices.append(ivertex)
					normals.append(inorm)
					texcoords.append(itcoord)
				mindices.append((nvs, dotex, donorm, j, len(vs)))
			self.indices.append((material, mindices))

		self.vbo_v = vbo.VBO(array(vertices, "f"))
		self.vbo_n = vbo.VBO(array(normals, "f"))
		self.vbo_t = vbo.VBO(array(texcoords, "f"))
		del self.vertices, self.normals, self.texcoords

	def bind(self):
		self.vbo_v.bind()
		glVertexPointerf(self.vbo_v)
		self.vbo_n.bind()
		glNormalPointerf(self.vbo_n)
		self.vbo_t.bind()
		glTexCoordPointerf(self.vbo_t)

	def basic_render(self):
		self.bind()
		glEnableClientState(GL_VERTEX_ARRAY)
		texon, normon = None, None
		for material, mindices in self.indices:
			self.mtl.bind(material)
			for nvs, dotex, donorm, ioffset, isize in mindices:
				if donorm != normon:
					normon = donorm
					(glEnableClientState if donorm else glDisableClientState)(GL_NORMAL_ARRAY)
				if dotex != texon:
					texon = dotex
					(glEnableClientState if dotex else glDisableClientState)(GL_TEXTURE_COORD_ARRAY)
				shape = [GL_TRIANGLES, GL_QUADS, GL_POLYGON][nvs-3]
				glDrawArrays(shape, ioffset, isize)

	def __del__(self):
		if self.vbo_v:
			self.vbo_v.delete()
			self.vbo_n.delete()
			self.vbo_t.delete()
		OBJ.__del__(self)

def LOG(img,i):
	if img is None:
		print ('img ',i, 'is None')

from collections import deque
class Render():
	crt_poses = deque([]) # correct pose
	buffer_glframes = deque([])
	current_frame = None
	gaussian = np.array([7, 26, 41, 26, 7])
	def __init__(self):
		self.Poses = []
		self.count = 0
		import pickle
		# self.crt_poses = None
		# self.crt_poses = pickle.load(open('newPoses.pickle', "rb"))

		from pygame import locals
		pygame.init()
		self.gl_width = 800
		self.gl_height = 600
		viewport = (self.gl_width, self.gl_height)

		self.srf = pygame.display.set_mode(viewport, locals.OPENGL | locals.DOUBLEBUF)

		glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
		glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
		glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
		glEnable(GL_LIGHT0)
		glEnable(GL_LIGHTING)
		glEnable(GL_COLOR_MATERIAL)
		glEnable(GL_DEPTH_TEST)
		glShadeModel(GL_SMOOTH)  # most obj files expect to be smooth-shaded

		# LOAD OBJECT AFTER PYGAME INIT
		sys.path.append('./dataset/3dmodel')
		self.obj = OBJ_vbo('./dataset/3dmodel/head.obj')

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		width, height = viewport
		gluPerspective(45.0, width / float(height), 0.01, 100.0)
		glEnable(GL_DEPTH_TEST)
		glMatrixMode(GL_MODELVIEW)

		self.rx, self.ry = (180, -90)
		self.tx, self.ty = (0, 0)

		self.FM = face_pose_class.FacePose('./models')

	def draw(self,im, index, frame_num):
		if im is not None:
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			glLoadIdentity()

			# RENDER OBJECT
			glTranslate(self.tx, self.ty, -1)
			glRotate(self.rx, 0, 1, 0)
			glRotate(self.ry, 1, 0, 0)
			# pose_name = ['Pitch', 'Yaw', 'Roll']     # respect to  ['head down','out of plane left','in plane right']

			Pitch, Yaw, Roll = self.FM.predictImage(im)

			if Pitch is not None:
				self.count += 1
				self.Poses.append([Pitch, Yaw, Roll])
				self.crt_poses.append([Pitch, Yaw, Roll])
				# A = np.array([0.,0.09,0.14])
				# O = np.array([0.,0.,0.])
				# O1 = np.array([0.,0.115,0.])
				# rotate,_ = cv2.Rodrigues(np.array(([Pitch, Roll, Yaw])))
				# AO1 = (A-O1)
				# AO1.shape = (1,3)
				# aa = AO1*rotate
				# bb = np.linalg.norm(np.array([-aa[0][2],-aa[1][2],-aa[2][2]]))
				# A1 = np.array([aa[0][2],aa[1][2],aa[2][2]])+(O1-O)
                #
				# normal = np.cross(A,A1)
				# costheta = np.dot(A,A1)*np.linalg.norm(A)*np.linalg.norm(A1)
				# #算出转轴和角度，只要旋转一次即可
				# glRotate(np.arccos(costheta), normal[0], normal[1], normal[2])
				r = (np.sqrt(5)+1)/2
				# self.pitch = Pitch*r
				# self.yaw = Yaw*r
				# self.roll = Roll*r
				if index < 4:
					if index < 2:
						pose = self.crt_poses[index]
						oglout = self.DrawTails(pose,r, index, frame_num)
						if oglout is not None:
							self.current_frame = oglout
						LOG(self.current_frame,1)
						return self.current_frame

					LOG(self.current_frame, 2)
					return self.current_frame
				elif index == frame_num - 2:
					for i in range(2):
						pose = self.crt_poses[4 - index]
						oglout = self.DrawTails(pose,r, index, frame_num)
						if oglout is not None:
							self.current_frame = oglout
						LOG(self.current_frame,3)
						return self.current_frame
				else:
					pitch5 = [self.crt_poses[i][0] for i in range(self.crt_poses.__len__())]
					yaw5 = [self.crt_poses[i][1] for i in range(self.crt_poses.__len__())]
					roll5 = [self.crt_poses[i][2] for i in range(self.crt_poses.__len__())]

					new_pitch = np.dot(self.gaussian, np.array(pitch5[0:5])) / 107
					new_yaw = np.dot(self.gaussian, np.array(yaw5[0:5])) / 107
					new_roll = np.dot(self.gaussian, np.array(roll5[0:5])) / 107

					new_pose = [new_pitch, new_yaw, new_roll]

					self.crt_poses.popleft()
					self.crt_poses[1] = new_pose

					oglout = self.DrawTails(new_pose, r, index, frame_num)
					if oglout is not None:
						self.current_frame = oglout

					LOG(self.current_frame, 4)
					return self.current_frame
				# if self.crt_poses is not None:
				# 	self.pitch = self.crt_poses[self.count][0] * r
				# 	self.yaw = self.crt_poses[self.count][1] * r
				# 	self.roll = self.crt_poses[self.count][2] * r
				# else:
				# 	self.pitch = Pitch * r
				# 	self.yaw = Yaw * r
				# 	self.roll = Roll * r
			else:
				LOG(self.current_frame, 6)
				return self.current_frame
		else:
			print('im in def draw(self,im): is None!')
			LOG(self.current_frame, 5)
			return self.current_frame

	def DrawTails(self, pose, r, index, frame_num):
		self.pitch = pose[0] * r
		self.yaw = pose[1] * r
		self.roll = pose[2] * r
# # Take Care, different from opengl coordinary
		glRotate(self.pitch, 1, 0, 0)  # x
		glRotate(self.roll, 0, 1, 0)  # y
		glRotate(self.yaw, 0, 0, 1)  # z

		glTranslate(0.1, -0.3, -0.2)
		glScale(2, 2, 2)

		self.obj.render()

		img_data_str = pygame.image.tostring(self.srf, 'RGB')
		img_data_str = np.fromstring(img_data_str, dtype=np.uint8)
		tmp = img_data_str.reshape(self.gl_height, self.gl_width, 3)
		oglout = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
		self.buffer_glframes.append(oglout)
		if index<2:
			self.buffer_glframes.popleft()
			pass
		elif index>3 and index<=frame_num-3:
			oglout = self.buffer_glframes.popleft()
		else:
			self.buffer_glframes.popleft()

		fname = "result.png"
		cv2.imwrite(fname, oglout)
		cv2.imshow('1', oglout)
		cv2.waitKey(1)

		if oglout is None:
			print ('oglout is None!!!')
		return oglout

def render():
	# from pygame.locals import *
	# from pygame.constants import *
	# from OpenGL.GLU import *
	pygame.init()
	gl_width = 800
	gl_height = 600
	viewport = (gl_width, gl_height)

	srf = pygame.display.set_mode(viewport, pygame.locals.OPENGL | pygame.locals.DOUBLEBUF)

	glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
	glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
	glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
	glEnable(GL_LIGHT0)
	glEnable(GL_LIGHTING)
	glEnable(GL_COLOR_MATERIAL)
	glEnable(GL_DEPTH_TEST)
	glShadeModel(GL_SMOOTH)  # most obj files expect to be smooth-shaded

	# LOAD OBJECT AFTER PYGAME INIT
	obj = OBJ_vbo('test1.obj')

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	width, height = viewport
	gluPerspective(45.0, width / float(height), 0.01, 100.0)
	glEnable(GL_DEPTH_TEST)
	glMatrixMode(GL_MODELVIEW)

	rx, ry = (180, -90)
	tx, ty = (0, 0)

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glLoadIdentity()

	# RENDER OBJECT
	glTranslate(tx, ty, -1)
	glRotate(rx, 0, 1, 0)
	glRotate(ry, 1, 0, 0)
	# pose_name = ['Pitch', 'Yaw', 'Roll']     # respect to  ['head down','out of plane left','in plane right']

	Pitch = -1.34
	Yaw = 10.49
	roll = -0.49

	# Take Care, different from opengl coordinary
	glRotate(Pitch, 1, 0, 0)  # x
	glRotate(roll, 0, 1, 0)  # y
	glRotate(Yaw, 0, 0, 1)  # z

	glScale(2, 2, 2)
	obj.render()
	pygame.display.flip()

	img_data_str = pygame.image.tostring(srf, 'RGB')
	img_data_str = np.fromstring(img_data_str, dtype=np.uint8)
	tmp = img_data_str.reshape(gl_height, gl_width, 3)
	oglout = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
	fname = "result.png"
	cv2.imwrite(fname, oglout)
	cv2.imshow('1', oglout)
	pass

# if __name__ == "__main__":
# 	# LMB + move: rotate
# 	# RMB + move: pan
# 	# Scroll wheel: zoom in/out
# 	import sys
# 	from pygame.locals import *
# 	from pygame.constants import *
# 	from OpenGL.GLU import *
#
# 	pygame.init()
# 	gl_width = 800
# 	gl_height = 600
# 	viewport = (gl_width,gl_height)
# 	hx = viewport[0]/2
# 	hy = viewport[1]/2
# 	srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
#
# 	glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
# 	glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
# 	glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
# 	glEnable(GL_LIGHT0)
# 	glEnable(GL_LIGHTING)
# 	glEnable(GL_COLOR_MATERIAL)
# 	glEnable(GL_DEPTH_TEST)
# 	glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded
#
# 	# LOAD OBJECT AFTER PYGAME INIT
# 	# obj = OBJ_vbo(sys.argv[1])
# 	obj = OBJ_vbo('test1.obj')
#
# 	clock = pygame.time.Clock()
#
# 	glMatrixMode(GL_PROJECTION)
# 	glLoadIdentity()
# 	width, height = viewport
# 	gluPerspective(45.0, width/float(height), 0.01, 100.0)
# 	glEnable(GL_DEPTH_TEST)
# 	glMatrixMode(GL_MODELVIEW)
#
# 	img_data_str = None
# 	# phi, theta, d = 0., 0., 10.
# 	rx, ry = (180,-90)
# 	tx, ty = (0,0)
# 	zpos = 1
# 	rotate = move = False
# 	playing = True
# 	jframe = tframe = 0
# 	while playing:
# 		dt = clock.tick() * 0.001
# 		tframe += dt
# 		jframe += 1
# 		for e in pygame.event.get():
# 			if e.type == QUIT:
# 				playing = False
# 			elif e.type == KEYDOWN and e.key == K_ESCAPE:
# 				playing = False
# 			elif e.type == KEYDOWN and e.key == K_SPACE:
# 				fname = "result.png"
# 				img_data_str = pygame.image.tostring(srf, 'RGB')
# 				img_data_str = np.fromstring(img_data_str,dtype=np.uint8)
# 				tmp = img_data_str.reshape(gl_height,gl_width,3)
# 				tmp = cv2.cvtColor(tmp,cv2.COLOR_RGB2BGR)
# 				cv2.imwrite(fname,tmp)
# 				cv2.imshow('1',tmp)
# 				cv2.waitKey()
# 				# img_data_str = pygame.surfarray.array2d(srf)
# 				print(img_data_str)
# 				print(type(img_data_str))
# 			elif e.type == MOUSEBUTTONDOWN:
# 				if e.button == 4: zpos = max(1, zpos-1)
# 				elif e.button == 5: zpos += 1
# 				elif e.button == 1: rotate = True
# 				elif e.button == 3: move = True
# 			elif e.type == MOUSEBUTTONUP:
# 				if e.button == 1: rotate = False
# 				elif e.button == 3: move = False
# 			elif e.type == MOUSEMOTION:
# 				i, j = e.rel
# 				if rotate:
# 					rx += i
# 					# ry += j
# 				if move:
# 					tx += i
# 					ty -= j
#
# 		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
# 		glLoadIdentity()
#
# 		# RENDER OBJECT
# 		glTranslate(tx, ty, -1)
# 		glRotate(rx, 0, 1, 0)
# 		glRotate(ry, 1, 0, 0)
# 		#pose_name = ['Pitch', 'Yaw', 'Roll']     # respect to  ['head down','out of plane left','in plane right']
# 		# roll = 4.33
# 		# Yaw = 22.07
# 		# Pitch = 2.29
# 		# roll = 1.56
# 		# Yaw = 16.57
# 		# Pitch = 16.89
#         #
# 		# roll = 2.20
# 		# Yaw = -8.68
# 		# Pitch = 13.39
#
# 		Pitch = -1.34
# 		Yaw = 10.49
# 		roll=-0.49
#
# 		# Pitch = -15.29
# 		# Yaw = 2.56
# 		# roll = 17.87
# 		# Take Care, different from opengl coordinary
# 		glRotate(Pitch,1,0,0) # x
# 		glRotate(roll,0,1,0) # y
# 		glRotate(Yaw,0,0,1)#z
#
# 		glScale(2,2,2)
# 		obj.render()
#
# 		pygame.display.flip()
# 	print ("%.2ffps" % (jframe / tframe))

