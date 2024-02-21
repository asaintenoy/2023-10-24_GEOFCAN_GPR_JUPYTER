from utils import *
from scipy.interpolate import griddata
from scipy.signal import medfilt2d, order_filter
from copy import copy
import pywt

def window(data, dx_dt, t0, tmax, trmin, trmax):
    """Fonction pour couper les ntr_max premières traces"""
    global data_w
    data_w = copy(data)
    dx, dt = dx_dt
    it0 = int(t0/dt)
    itmax = int(tmax/dt)
    data_w = data[it0:itmax, trmin:trmax]
    x = np.arange(trmax-trmin)*dx
    t = np.arange(itmax-it0)*dt
    nsw = itmax-it0
    ntrw = trmax-trmin
    return data_w, t, x, nsw, ntrw

def mean_tr_rm(data):
    """Fonction retrait de la trace moyenne sur un radargramme"""
    mean_tr = np.mean(data, axis=1)
    global data_m
    data_m = copy(data)
    ns, ntr = data.shape
    for n in range(ntr):
        data_m[:,n] = data[:,n] - mean_tr
    return data_m

def median_tr_rm(data):
    """Fonction retrait de la trace mediane sur un radargramme"""
    global data_m
    data_m = copy(data)
    ns, ntr = data.shape
    for n in range(ntr):
        data_m[:,n] = data[:,n] - np.median(data,axis=1)
    return data_m

def FT(trace,ns,dt):
    """Fast Fourier transform"""
    trace_fft = np.fft.fft(trace)
    df = 1/(ns*dt) # en gigaHertz
    return trace_fft, df

def TO(data, coef_range, wavelet):
    """Transformée en ondelettes avec l'ondelette wavelet (string)"""
    coefs, freqs = pywt.cwt(data, coef_range, wavelet)
    return coefs, freqs

def inv_to(coefs, wavelet):  
    """ Inverse transfom of wavelet analysis"""
    mwf = pywt.ContinuousWavelet(wavelet).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]
    r_sum = np.transpose(np.sum(np.transpose(coefs), axis=-1))
    return r_sum * (1 / y_0)

def coupe_coef(coef, seuils):
    """Coupe les coefs obtenus lors d'une TO sur une trace entre différents seuils"""
    S,T = np.shape(coef)
    c = coef
    S_bas, S_haut, a, b = seuils
    print("seuils", seuils)
    for i in np.arange(S):
        for j in np.arange(T):
            if i > S_haut:
                c[i,j] = 0
            if i < S_bas:
                c[i,j] = 0
            if c[i,j]>a or c[i,j]< -a or (c[i,j] < b and c[i,j] > 0) or (c[i,j] > -b and c[i,j] < 0):
                c[i,j]=0
    return c

def filt_SVD(data, ns, ntr, SVmin, SVmax):  
    """ SVD filtering"""
    image = data.reshape(ns,ntr)
    # SVD computation
    U, sigma, V = np.linalg.svd(image)
    # Reconstitution keeping SV between SVmin and SVmax
    reconstimg = np.matrix(U[:, SVmin:SVmax]) * np.diag(sigma[SVmin:SVmax]) * np.matrix(V[SVmin:SVmax, :])
    return reconstimg
		
def dc_removal(data, dx_dt, window_width):
    """ Remove the DC bias by using a moving mean filter. window_width (ns)"""
    dx, dt = dx_dt
    samples, traces = data.shape
    t = np.linspace(0, 1, samples) * (dt * samples)
    data_dc = copy(data)
    width = int(window_width / dt)
    half_width = int(width / 2)
    temp = np.ones((samples, traces))
    temp *= data
    for trace in range(traces):
        for index in range(samples):
            if index < half_width:
                data_dc[index,trace] += -np.mean(temp[:index + half_width, trace])
            elif index > (samples - half_width):
                data_dc[index,trace] += -np.mean(temp[index-half_width:])
            else:
                 data_dc[index,trace] += -np.mean(temp[index-half_width:index+half_width, trace])
    return data_dc

def dc_substraction(data, dx_dt, time_window):
    """ Remove the DC bias by using a simple substraction of the DC offset
    time_window is a tuple (start, stop) in ns."""
    data_dc = copy(data)
    dx, dt = dx_dt
    start, stop = time_window
    istart = int(start/dt)
    istop = int(stop/dt)
    data_dc = data - np.mean(data[istart:istop])
    return data_dc

def cut_off_frequency(data, dx_dt, fc):

    """ Apply a pass-band filter by using Fast Fourier Transform. fc (MHz) must be below the bandwidth of the recorded data  ~ 10 MHz """
    dx, dt = dx_dt
    samples, traces = data.shape
    f = np.linspace(0, 1, samples) * (samples / dt)
    index = int(fc * dt)
    fftdata = np.fft.fft2(data)
    for trace in range(traces):
        fit = np.diff(fftdata.real[index:index+2, trace]) * [range(index)]
        fftdata.real[:index, trace] = fit
    data = np.fft.ifft2(fftdata)
    return data.real

def time_zero(data, dx_dt, t0 = 0.0):

    """Replaces the start time of your radargrams by t0 (ns), retrun a new 2D dataset reshaped"""

    dx, dt = dx_dt
    samples, traces = data.shape
    t = np.linspace(0, 1, samples) * (samples * dt)
    index = int(t0 / dt)
    return data[index:]

def user_gain (data, dx_dt, sort, param, time_window, plot=False, return_fgain=False):
	
	"""Add a user defined gain chosen beetween {'constant', 'linear', 'enxponential'} on data.
	param is a tuple (a, b) where --> constant : fgain = a, linear : fgain = a*t, exponential : fgain = a*exp(b*t)
	time_window is a tuple (start, stop) in ns.
	"""
	dx, dt = dx_dt
	samples, traces = data.shape
	data_g = copy(data)
	a, b = param
	t = np.linspace(0, 1, samples) * (samples * dt)
	t0, stop = time_window
	
	start = int(t0 / dt)-1
	stop = int(stop / dt)
	width = start-stop
	fgain = np.ones(samples)
	
	if sort == 'constant':
		fgain[start:stop] = [a]*width
		
	elif sort == 'linear':
		fgain[start:stop] = [a*(t-t0) + 1 for t in t[start:stop]]

	elif sort == 'exponential':
		fgain[start:stop] = [a*(np.exp(b*(t-t0)) - 1) + 1 for t in t[start:stop]]
		
	for trace in range(traces):
		data_g[:, trace] = data[:, trace] * fgain.astype(dtype=data.dtype)
	
	if plot is True:
		plt.plot(fgain)
		plt.show()
	
	if return_fgain:
		return data_g, fgain
	return data_g

def velocity_analysis(data, dx_dt, param, clip, num_profil, impression):
    """
    Plot the radargram along with the hyperbolic function initialized by a tuple = (x0 (m), t0 (ns), c (m/ns), r (m)).
    """

    samples, traces = data.shape
    dx, dt = dx_dt
    x0, t0, v, r, width = param

    mid = int(x0 / dx)
    start = int(mid - width / (2 * dx))
    stop = int(mid + width / (2 * dx))
    
    z0 = (t0 * v + 2 * r) / 2
    x = np.linspace(0, 1, traces) * (traces * dx)
    t = np.linspace(0, 1, samples) * (samples * dt)
    hyperbol =  (2 / v) * (np.sqrt((x0-x[start:stop])**2 + z0**2) - r) 
    
    fig, ax = plt.subplots(num='velocity_analysis', figsize=(20, 6), facecolor='w', edgecolor='w')
    ims = ax.imshow(data, extent=[np.amin(x), np.amax(x), np.amax(t), np.amin(t)], interpolation='nearest', aspect='auto',
                    cmap='seismic', vmin=-np.amax(abs(data)*clip), vmax=np.amax(abs(data)*clip))
    ax.plot(x[start:stop], hyperbol,linewidth=5,c='orange',linestyle='dashed')
    if impression == True:
        fig.savefig('radargramme_hyp' + str(num_profil) + '.png', bbox_inches='tight', dpi=150)
    plt.grid()
    cbar = fig.colorbar(ims)
    cbar.ax.set_ylabel('Amplitude')

def stolt_migration(data, dx_dt, c):
	
	# Imput parameters
	dx, dt = dx_dt
	fs = 1/dt
	eps = 2.2e-16
	nt0, nx0 = data.shape
	t = np.linspace(0,nt0*dt,nt0) 
	x = np.linspace(0,nx0*dx,nx0)
	
	# Zero-padding 2D
	nt = 2 * nextpower(nt0)
	nx = 2 * nx0
	
	# One Emiter-Receiver scenario
	ERMv = c / 2
	
	# FFT & shift 
	fftdata = np.fft.fftshift(np.fft.fft2(data, s=(nt,nx)))
	
	# Build (kx, f) 
	f = np.linspace(-nt/2, nt/2-1, nt) * fs / nt 
	kx = np.linspace(-nx/2,nx/2-1, nx) / dx / nx
	kx, f = np.meshgrid(kx, f)
	
	# Remove evanescent parts
	evanescent = (abs(f)  / abs(kx+eps) > c).astype(int)
	fftdata *= evanescent
	
	# Stolt remapping function f(kz)
	fkz = ERMv*np.sign(f)*np.sqrt(kx**2 + f**2/ERMv**2)
	
	# Linear interpolation on grid
	fftdata = griddata((kx.ravel(), f.ravel()), fftdata.ravel(), (kx, fkz), method='nearest')
	
	# Jacombien
	fftdata *= f / (np.sqrt(kx**2 + f**2/ERMv**2)+eps)
	
	# IFFT & Migrated data 
	mig = np.fft.ifft2(np.fft.ifftshift(fftdata))
	mig = mig[:nt0,:nx0]
	
	dz = dt * c / 2
	dx_dz = (dx, dz)
	
	return abs(mig), dx_dz
	
	
	
	
