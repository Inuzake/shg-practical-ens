import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

# ANHARMONIC PARAMETERS
A, B, C = 1, 1, -2

# HARMONIC PARAMETERS
# A, B, C = 0, 0, 1



def run_fft_analysis(x, t, dt):
    """
    FFT ANALYSIS
    ============
    """
    # Perform FFT
    n = len(x)
    # print(len(x), len(t))
    fft_result = np.fft.fft(x)
    # Frequency axis
    frequencies = np.fft.fftfreq(n, dt)  
    
    # Take only positive frequencies
    fft_magnitude = np.abs(fft_result) / n
    positive_freqs = frequencies[:n//2]
    # multiply by 2 (single-sided spectrum)
    positive_magnitude = 2 * fft_magnitude[:n//2]  
    
    # Time-domain signal
    plt.plot(t, x, color = 'k')
    plt.title("Time Domain Signal")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.show()
    plt.clf()
    
    # Frequency-domain (FFT)
    # Plot the x times 2 pi
    omega = positive_freqs * 2 * np.pi
    plt.plot(omega,  omega**2 * positive_magnitude, label = 'FFT', color = 'k', lw = 3)
    plt.xlim(0.0, 5)
    # plt.ylim(0, 0.1)
    # plt.axvline(1 /(np.pi * 2))
    plt.title("Frequency Domain (FFT)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.show()


def rk_step(x, v, t, dt, E = 0.0, omega = 0.0):
    """
    RK STEP
    =======
    """

    dx1 = dt * v
    dv1 = dt * force(x, t, E = E, omega = omega)


    dx2 = dt * (v + dv1 * 0.5)
    dv2 = dt * force(x + 0.5 * dx1, t + 0.5 * dt, E = E, omega = omega)


    dx3 = dt * (v + dv2 * 0.5)
    dv3 = dt * force(x + 0.5 * dx2, t + 0.5 * dt, E = E, omega = omega)


    dx4 = dt * (v + dv3)
    dv4 = dt * force(x + dx3, t + dt, E = E, omega = omega)

    
    return x + (dx1 + 2 * dx2 + 2 * dx3 + dx4)/6, v + (dv1 + 2 * dv2 + 2 * dv3 + dv4)/6

def force(x, t, E = 0.0, omega = 0.0):
    """
    GET THE FORCE ON THE ELECTRON
    =============================
    """
    f = - A * x**3 - B * x**2 - C * x

    f -= E * np.sin(omega * t)

    return f

if __name__ == '__main__':
    """
    SHG 

    Comment 

    If you run the code with the following parameters 

        A, B, C = 1, 1, -2

    and

        E0, OMEGA0 = 0.1, 1
        
    you will see in the final plot of the script the following peaks

    1) omega = OMEGA0 which correspond to the first (trivial) harmonic.

    2) omega = 2 OMEGA0  which correspond to the second harmonic (SHG) !!!

    3) omega = OMEGA_norm = 2.5 corresponds to the normal mode of the potential.
    You can compute the normal mode frequency using the harmonic approximation.

    4) omega = 3.5 which correponds to the combination mode of the normal mode with the first harmonic
       Indeed 3.5 = OMEGA_norm + OMEGA0.
    """
    # This ensures the script always runs from its own folder, no matter where you launch it from
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    x0, v0, E0, OMEGA0 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

    dt = 0.01
    Nsteps = 100000


    print("\n\n============== RK4 SHG ==============")
    print('POTENTIAL')
    print('A {} B {} C {}'.format(A, B, C))
    print('E0 {} OMEGA0 {}'.format(E0, OMEGA0))
    print('INITIAL CONDITIONS')
    print('x0 {} v0 {}'.format(x0, v0))
    print('STEPS AND dt')
    print('N {} dt {}'.format(Nsteps, dt))
    print()
    
    all_x, all_v, all_t = [], [], []
    all_x.append(x0)
    all_v.append(v0)
    all_t.append(0.0)
    
    for i in range(Nsteps):
        xnew, vnew = 0.0, 0.0
        xnew, vnew = rk_step(all_x[-1], all_v[-1], all_t[-1], dt, E = E0, omega = OMEGA0)
        all_x.append(xnew)
        all_v.append(vnew)
        all_t.append(i * dt)
    
    run_fft_analysis(np.asarray(all_x), np.asarray(all_t), dt)