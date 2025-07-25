import mokka.channels.torch as channels
import mokka.utils as utils
import torch

num_samples = 1000

symbolrate = 30e9
oversampling = 16
dt = 1 / (oversampling * symbolrate)
dz = 1  # km
wavelength = 1550  # nm
span_length = length_span = 100  # km
num_span = 10
amp_gain = "alpha_equalization"
alphadb = alphaa_db = alphab_db = 0.2  # db/km
D = 16  # [ps/nm/km]
gamma = torch.tensor(1.3)  # [1/W/km]
betaa2 = utils.beta2(D, wavelength)
betapa = betap = torch.tensor([0, 0, betaa2])
betab2 = betaa2
betapb = torch.tensor([0, 0, betab2])
amp_noise = True
noise_figure = 5  # dB
optical_carrier_frequency = wavelength / 0.3  # GHz
bw = 2 * symbolrate
P_input = 0  # dBm
P_input_lin = 10 ** ((P_input - 30) / 10)  # W
padding = 1000


def test_awgn():
    channel = channels.ComplexAWGN()
    sigma = torch.tensor(0.1)
    x = torch.zeros(num_samples, dtype=torch.complex64)
    y = channel(x, sigma)
    assert len(y) == num_samples


def test_phasenoise():
    channel = channels.PhasenoiseWiener()
    sigma_phi = torch.tensor(0.1)
    sigma_n = torch.tensor(0.01)
    x = torch.zeros(num_samples, dtype=torch.complex64)
    y = channel(x, sigma_phi, sigma_n)
    assert len(y) == num_samples


def test_residualphasenoise():
    channel = channels.ResidualPhaseNoise()
    sigma_phi = torch.tensor(0.1)
    x = torch.zeros(num_samples, dtype=torch.complex64)
    y = channel.forward(x, sigma_phi)
    assert len(y) == num_samples


def test_opticalnoise():
    wavelength_bw = 0.1  # nm
    optical_snr = 25  # dB
    channel = channels.OpticalNoise(dt, wavelength, wavelength_bw, optical_snr)
    x = torch.zeros(num_samples, dtype=torch.complex64)
    y = channel(x)
    assert len(y) == num_samples


def test_edfa_singlepol():
    amp = channels.EDFAAmpSinglePol(
        span_length,
        amp_gain,
        alphadb,
        amp_noise,
        noise_figure,
        optical_carrier_frequency,
        bw,
        P_input_lin,
        padding,
    )
    x = torch.zeros(num_samples, dtype=torch.complex64)
    y = amp(x, 0)
    assert len(y) == num_samples
    y = amp(x, 1)
    assert len(y) == num_samples
    y = amp(x, 2)
    assert len(y) == num_samples


def test_SSFM_singlepol():
    amp = channels.EDFAAmpSinglePol(
        span_length,
        amp_gain,
        alphadb,
        amp_noise,
        noise_figure,
        optical_carrier_frequency,
        bw,
        P_input_lin,
        padding,
    )
    channel = channels.SSFMPropagationSinglePol(
        dt, dz, alphadb, betap, gamma, length_span, num_span, amp=amp
    )
    x = torch.zeros(num_samples, dtype=torch.complex64)
    y = channel(x)
    assert len(y) == num_samples


def test_SSFM_dualpol():
    amp = channels.EDFAAmpDualPol(  # noqa
        span_length,
        amp_gain,
        alphaa_db,
        alphab_db,
        amp_noise,
        noise_figure,
        optical_carrier_frequency,
        bw,
        P_input_lin,
        padding,
    )
    channel = channels.SSFMPropagationDualPol(
        dt, dz, alphaa_db, alphab_db, betapa, betapb, gamma, length_span, num_span
    )
    x1 = torch.zeros(num_samples, dtype=torch.complex64)
    x2 = torch.zeros(num_samples, dtype=torch.complex64)
    y1, y2 = channel(x1, x2)
    assert len(y1) == num_samples
    assert len(y2) == num_samples


def test_SSFM_dualpol_circular():
    amp = channels.EDFAAmpDualPol(
        span_length,
        amp_gain,
        alphaa_db,
        alphaa_db,
        amp_noise,
        noise_figure,
        optical_carrier_frequency,
        bw,
        P_input_lin,
        padding,
    )
    channel = channels.SSFMPropagationDualPol(
        dt,
        dz,
        alphaa_db,
        alphaa_db,
        betapa,
        betapb,
        gamma,
        length_span,
        num_span,
        solution_method="circular",
        amp=amp,
    )
    x1 = torch.zeros(num_samples, dtype=torch.complex64)
    x2 = torch.zeros(num_samples, dtype=torch.complex64)
    y1, y2 = channel(x1, x2)
    assert len(y1) == num_samples
    assert len(y2) == num_samples
