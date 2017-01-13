import os

import numpy as np
import matplotlib.pyplot as plt

from lasp.sound import plot_spectrogram, WavFile, play_sound
from lasp.timefreq import gaussian_stft


class TwoMassOscillator(object):


    def __init__(self, m=0.250, k0=1e6, k12=30, eta=1e-5):

        self.m = m
        self.k0 = k0
        self.k12 = k12
        self.eta = eta
        self.beta = self.eta*2*np.sqrt(self.k0 / self.m)

        self.A = np.zeros([4, 4])

        self.x = np.zeros([4])
        self.u = 0.

        self.set_transition_matrix(0.)

    def set_transition_matrix(self, dk):

        a = (self.k0 + dk + self.k12) / self.m
        b = self.beta / self.m
        c = self.k12 / self.m

        self.A[0, 1] = 1.

        self.A[1, 0] = -a
        self.A[1, 1] = -b
        self.A[1, 2] = c

        self.A[2, 3] = 1.

        self.A[3, 0] = c
        self.A[3, 2] = -a
        self.A[3, 3] = -b

    def step_rk4(self, dk, u, dt):

        self.set_transition_matrix(dk)
        uv = np.array([0, u, 0, u])
        dt2 = dt / 2.
        dt6 = dt / 6.

        rk1 = np.dot(self.A, self.x) + uv
        rk2 = np.dot(self.A, self.x + dt2*rk1) + uv
        rk3 = np.dot(self.A, self.x + dt2*rk2) + uv
        rk4 = np.dot(self.A, self.x + dt*rk3) + uv

        self.x = self.x + dt6*(rk1 + 2*rk2 + 2*rk3 + rk4)


def write_and_play(wave, sample_rate):

    wf = WavFile()
    wf.sample_rate = sample_rate
    wf.data = wave
    wf.num_channels = 1

    fname = '/tmp/temp_wavfile_12345.wav'
    wf.to_wav(fname, normalize=True)
    play_sound(fname)
    os.remove(fname)


def simulate_2mass_pulse(duration, flow_magnitude=0.1, dt=20e-6):

    sample_rate = 1. / dt
    cord_model = TwoMassOscillator()
    nt = int(duration / dt)
    x = list()
    full_t = np.arange(nt)*dt
    dk = np.abs(np.arange(nt)*1e2 * np.cos(2*np.pi*full_t))
    for t in range(nt):
        cord_model.step_rk4(dk[t], flow_magnitude, dt)
        x.append(cord_model.x)

    x = np.array(x)

    t = np.arange(nt)*dt
    plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(t, x[:, 0], 'k-')
    # plt.plot(t, x[:, 1], 'r-')

    # plt.subplot(3, 1, 2)
    # plt.plot(t, x[:, 2], 'k-')
    # plt.plot(t, x[:, 3], 'r-')

    ax = plt.subplot(1, 1, 1)
    spec_t,spec_freq,spec,rms = gaussian_stft(x[:, 0], sample_rate, 20e-3, 1e-3, min_freq=0, max_freq=1e3)
    plot_spectrogram(spec_t, spec_freq, spec, ax=ax)
    plt.axis('tight')

    wave = x[:, 0]
    write_and_play(wave, sample_rate)

    plt.show()


if __name__ == '__main__':

    simulate_2mass_pulse(10.00, flow_magnitude=200, dt=20e-6)


