import scipy, scipy.fft, scipy.signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_traj(trajectory):
    """
    PLOT TRAJECTORY
    ===============
    """
    labels = ["x(t)", "v(t)", "V(t)", "K(t)", "E(t)"]
    colors = ["red", "green", "purple", "orange", "blue"]
    # Create the figure
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(3, 2, figure=fig)  
    # Create subplots with custom positions
    axs = []
    for i in range(3):
        for j in range(2):
            if i == 2:
                ax = fig.add_subplot(gs[i, :])
                axs.append(ax)
                break
            else:
                ax = fig.add_subplot(gs[i, j])
                axs.append(ax)


    for i in range(len(axs)):
        axs[i].plot(trajectory[:,0], trajectory[:,i+1], color = colors[i],  lw = 3)
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel("t")

    plt.tight_layout()
    plt.show()
    plt.close()


def get_green_function(x, t, smooth = True):
    """
    GET THE GREEN FUNCTION
    ======================
    """
    # The len of the signal
    n = len(x)
    
    x_padded = np.zeros(2 * n, dtype = type(x))
        
    # Set the inital values and the others will be zeros
    x_padded[:n] = x
    
    # The P P correlation in Fourier
    x_omega  = scipy.fft.fft(x_padded)
    
    # Get the modulus squre
    x2_omega = np.abs(x_omega)**2
    
    # Perform the inverse Fourier transform
    C_t = scipy.fft.ifft(x2_omega)
    
    # Get the time correlation function
    normalization = np.arange(n, 0, -1)
    C_t_real = np.real(C_t)[:n] /normalization

    
    # Now we want the Fourier representation of <x(t) x(0)>
    C_t_real_padded = np.zeros(2 * n, dtype = type(C_t_real[0]))
    C_t_real_padded[:n] = C_t_real
    C_t_real_padded[0] = 0.5 * C_t_real_padded[0]
    # Get the Fourier transform
    if smooth:
        C_omega = scipy.fft.fft(C_t_real_padded * scipy.signal.windows.hann(2 * n))
    else:
        C_omega = scipy.fft.fft(C_t_real_padded)
        
    # Get the frequencies
    frequencies = np.fft.fftfreq(2 * n, t[1] - t[0]) 
    positive_freqs = frequencies[:n]

    omega = positive_freqs * 2 * np.pi
    plt.plot(omega, omega**2 * np.abs(C_omega[:n])**2)
    plt.xlim(0.1,10)
    plt.tight_layout()
    plt.show()
    plt.close()

def fft_analysis(x, t):
    """
    FFT ANALYSIS
    ============
    """
    # Perform FFT
    n = len(x)
    fft_result = np.fft.fft(x)
    frequencies = np.fft.fftfreq(n, t[1] - t[0])  # Frequency axis
    
    # Take only positive frequencies
    fft_magnitude = np.abs(fft_result) / n
    positive_freqs = frequencies[:n//2]
    positive_magnitude = 2 * fft_magnitude[:n//2]  # multiply by 2 (single-sided spectrum)
    
    # Plot the signal and FFT
    plt.figure(figsize=(12, 5))
    
    # Time-domain signal
    plt.subplot(1, 2, 1)
    plt.plot(t, x)
    plt.title("Time Domain Signal")
    plt.xlabel("Time")
    plt.ylabel("Position")
    
    # Frequency-domain (FFT)
    plt.subplot(1, 2, 2)
    # Plot the x times 2 pi
    omega = positive_freqs * 2 * np.pi
    plt.plot(omega, omega**2 * positive_magnitude)
    plt.xlim(0.1,10)
    plt.ylim(0, 0.1)
    # plt.axvline(1 /(np.pi * 2))
    plt.title("Frequency Domain (FFT)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    plt.close()


def fft_analysis_new(x, t):
    """
    FFT ANALYSIS
    ============
    
    Padding
    """
    # Perform FFT after padding
    n = len(x)
    x_padded = np.zeros(2 * n, dtype = type(x[0]))
    x_padded[:n] = x

    fft_result = scipy.fft.fft(x_padded)
    freq = scipy.fft.fftfreq(2 * n, t[1] - t[0])

 
    # e frequencies
    fft_magnitude = np.abs(fft_result) / n
    positive_freqs = freq[: (2 * n) //2]
    positive_magnitude = 2 * fft_magnitude[: (2 * n) //2] 
    
    # Plot the signal and FFT
    plt.figure(figsize=(12, 5))
    
    # Time-domain signal
    plt.subplot(1, 2, 1)
    plt.plot(t, x)
    plt.title("Time Domain Signal")
    plt.xlabel("Time")
    plt.ylabel("Position")
    
    # Frequency-domain (FFT)
    plt.subplot(1, 2, 2)
    # Plot the x times 2 pi
    omega = positive_freqs * 2 * np.pi
    plt.plot(omega,  omega**2 * positive_magnitude**2)
    plt.xlim(0.1,3)
    # plt.ylim(0, 0.1)
    # plt.axvline(1 /(np.pi * 2))
    plt.title("Frequency Domain (FFT)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    plt.close()



# Read the trajectory
trajectory = np.loadtxt("traj.txt")
plot_traj(trajectory)

# Get positions and times
pos = trajectory[:,1]
times = trajectory[:,0]

# NO PADDING
fft_analysis(pos, times)

# PADDING THE SIGNAL
# fft_analysis_new(pos, times)

# STUDY THE GREEN FUNCTION
# get_green_function(pos, times)
