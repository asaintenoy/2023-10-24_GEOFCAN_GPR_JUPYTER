import sys, h5py, binascii
import numpy as np
import matplotlib.pyplot as plt


def plot_radargram(data, dx_dt, clip):
    """ Plot B-scan and save in file name_fig.png if fig = True """
    dx, dt = dx_dt
    cmap = 'seismic'
    samples, traces = data.shape
    t = np.linspace(0, 1, samples) * (samples * dt)
    x = np.linspace(0, 1, traces) * (traces * dx)
    fig = plt.figure(figsize=(20, 6))
    pos=plt.imshow(data, extent=[np.amin(x), np.amax(x), np.amax(t), np.amin(t)], interpolation='nearest', aspect='auto', cmap=cmap, vmin=-np.amax(clip*abs(data)), vmax=np.amax(clip*abs(data)))
    fig.colorbar(pos)
    plt.xlabel('Distance [m]')
    plt.ylabel('Two-way travel time [ns]')
    plt.grid()
        
def image_radargram(data, dx_dt, clip, boolean, name_fig, cmap):
    """ Plot B-scan and save in file name_fig.png if fig = True """
    dx, dt = dx_dt
    cmap = cmap
    samples, traces = data.shape
    t = np.linspace(0, 1, samples) * (samples * dt)
    x = np.linspace(0, 1, traces) * (traces * dx)
    fig = plt.figure(figsize=(20, 10))
    pos=plt.imshow(data, extent=[np.amin(x), np.amax(x), np.amax(t), np.amin(t)], interpolation='nearest', aspect='auto', cmap=cmap, vmin=-np.amax(clip*abs(data)), vmax=np.amax(clip*abs(data)))    
    plt.xticks([]) 
    plt.yticks([]) 
    if boolean == True :
        print(name_fig)
        plt.tight_layout()
        plt.savefig(name_fig + '.png', bbox_inches='tight', dpi=300)
    
def plot_datamig (data, dx_dt, v, clip, num_profil, impression):
	
	""" Plot B-scan """
	
	dx, dt = dx_dt
	samples, traces = data.shape
	t = np.linspace(0, 1, samples) * (samples * dt)
	x = np.linspace(0, 1, traces) * (traces * dx)
	fig = plt.figure(num=str(v), figsize=(20, 10), facecolor='w', edgecolor='w')
	pos=plt.imshow(data, extent=[np.amin(x), np.amax(x), np.amax(t), np.amin(t)], interpolation='nearest', aspect='auto', cmap='seismic', vmin=-np.amax(clip*abs(data)), vmax=np.amax(clip*abs(data)))
	fig.colorbar(pos)
	plt.xlabel('Distance [m]')
	plt.ylabel('Depth [m]')
	plt.grid()
	if impression == True:
		fig.savefig('radargramme_mig' + str(num_profil) + '_vmig_' + str(np.round(v,2)) + '.png',bbox_inches='tight',dpi=150)    
	plt.show()    
    
    
    
	
def plot_colorbar (data, dx_dt):
	
	""" Plot B-scan colorbar """
	dx, dt = dx_dt
	samples, traces = data.shape
	t = np.linspace(0, 1, samples) * (samples * dt)
	x = np.linspace(0, 1, traces) * (traces * dx)
	fig, ax = plt.subplots()
	pos = plt.imshow(data, extent=[np.amin(x), np.amax(x), np.amax(t), np.amin(t)], interpolation='nearest', aspect='auto', cmap='seismic', vmin=-np.amax(abs(data)), vmax=np.amax(abs(data)))
	fig.colorbar(pos,ax)
	
	plt.show()    
def nextpower (n, base = 2.0):
    """Return the next integral power of two greater than the given number.
    Specifically, return m such that
        m >= n
        m == 2**x
    where x is an integer. Use base argument to specify a base other than 2.
    This is useful for ensuring fast FFT sizes.
    """
    x = base**np.ceil(np.log(n) / np.log(base))
    if type(n) == np.ndarray:
        return np.asarray (x, dtype=int)
    else:
        return int (x)

def read_ascii (filename):
	"""
	Comments
	"""
	f = open(filename, 'r')
	mylist = f.readlines()
	while '\n' in mylist: mylist.remove('\n')
	
	# Format data
	traces = np.shape(mylist)[0]
	data = []
	for trace in range(traces):
		data.append(list(map(int, mylist[trace].split())))
	f.close()
	
	return np.array(data).T

def read_hdf5 (filename):
	""" Convert a 1 receiver h5py data file into ascii file. """
	
	f = h5py.File(filename, 'r')
	path = '/rxs/rx1/'
	modelruns = f.attrs['Modelruns']
	samples = f.attrs['Iterations'] 
	dt = f.attrs['dt']*1e9
	positions = f.attrs['Positions'][:,0,0]
	dx = np.diff(positions)[0]
	data = np.ones((samples, modelruns))
	for model in range(modelruns):
		data[:,model] = f['%s%s' % (path, 'Ez')][:,model]

	return data, (dx, dt)

def read_ramac (filename):
	
	""" Read .rad and .rd3 files and retrun a 2D dataset along with steps (dx (m), dt(ns)) """
	
	# Read header
	f = open('%s%s' %(filename, '.rad'), 'r')
	mylist = f.read().split('\n')
	f.close()
	
	samples = int(mylist[0].split(':')[1])
	frequency = float(mylist[1].split(':')[1])*1e-3
	traces = int(mylist[22].split(':')[1])
	distance = float(mylist[23].split(':')[1])
	
	dt = 1 / frequency
	dx = distance / traces
	
	# Read data
	f = open('%s%s' % (filename, '.rd3'), 'rb')
	mylist = f.read()

	data = np.array(np.fromstring(mylist, dtype= 'int16'))
	traces = int(np.shape(data)[0] / samples)
	data = data.reshape((traces, samples))
	
	return data.T, (dx,dt)

def save_new_version (data, filename):
	
	f.open(filename, 'wb')
	np.save(f, data)
