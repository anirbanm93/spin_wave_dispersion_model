import numpy as np

class DispersionModel:
    """
        Description:
            This module implements the Kalinikos-Slavin spin-wave dispersion model
            for ferromagnetic thin films, as described in:

            Kalinikos, B. A., and Slavin, A., N. 
            "Theory of dipole-exchange spin wave spectrum for ferromagnetic films 
            with mixed exchange boundary conditions." 
            Journal of Physics C: Solid State Physics 19.35 (1986): 7013-7033.
            DOI: 10.1088/0022-3719/19/35/014

        Inputs:
            film_prop : dict
                {
                    'Ms': Saturation magnetization (A/m),
                    'd':  Film thickness (m),
                    'Aex': Exchange stiffness constant (J/m)
                }

            kwargs : dict
                {
                    'mode_no'     : int     → Mode number n
                    'ksw'         : float   → In-plane spin-wave wavenumber (rad/m)
                    'Heff'        : float   → Effective magnetic field (A/m)
                    'config'      : str | tuple → Propagation configuration: 
                                                'MSSW', 'BVSW', 'FVSW', or (theta, phi)
                    'pinning_cond': bool    → True if surface spins are totally pinned
                }

        Outputs:
            est_freq() : Returns the estimated spin-wave frequency in Hz
            F_nn()     : Returns the dipole-exchange correction term (dimensionless)

        Reference Equation Numbers:
            Eq. (A10), (A12), (A14), (45), and (46) — Kalinikos & Slavin (1986)
    """

    def __init__(self, film_prop, **kwargs):
        # --- Universal constants ---
        self.gammaLL = 1.76e11                 # gyromagnetic ratio (rad/T·s)
        self.mu0 = 4e-07 * np.pi               # permeability of free space (H/m)

        # --- Film properties ---
        required_keys = ('Ms', 'd', 'Aex')
        if not all(k in film_prop for k in required_keys):
            raise KeyError(f"film_prop must include {required_keys}.")
        self.Ms = film_prop['Ms']
        self.d = film_prop['d']
        self.Aex = film_prop['Aex']

        # --- Derived quantities ---
        self.Lambdaex = (2 * self.Aex) / (self.mu0 * self.Ms ** 2)  # exchange length squared

        # --- Wave number and mode ---
        self.n = kwargs.get('mode_no', 0)
        self.ksw = kwargs.get('ksw', [0.0])
        if self.ksw.min() == 0:
            raise ValueError("ksw (in-plane wave number) cannot be zero.")
        self.kn = (self.n * np.pi) / self.d
        self.ktot = np.sqrt(self.ksw ** 2 + self.kn ** 2)

        # --- Magnetic field and frequency scale ---
        self.OmgH = kwargs.get('Heff', 0.0) / self.Ms  # normalized field
        if self.OmgH == 0:
            raise ValueError("Ratio between effective field and saturation magnetization cannot be zero.")
        self.fm = (self.gammaLL * self.mu0 * self.Ms) / (2 * np.pi)  # frequency scaling factor (Hz)

        # --- Geometry configuration ---
        config = kwargs.get('config', 'MSSW')
        if isinstance(config, str):
            if config == 'MSSW':
                self.theta, self.phi = np.pi / 2, np.pi / 2
            elif config == 'BVSW':
                self.theta, self.phi = np.pi / 2, 0
            elif config == 'FVSW':
                self.theta, self.phi = 0, 0
            else:
                raise ValueError("config must be one of {'MSSW', 'BVSW', 'FVSW'}.")
        elif isinstance(config, (tuple, list)) and len(config) == 2:
            self.theta, self.phi = config
        else:
            raise ValueError("Invalid config format. Must be a string or (theta, phi) tuple.")

        # --- Surface spin boundary condition ---
        self.pinning_cond = bool(kwargs.get('pinning_cond', False))

    # -------------------------------------------------------------------------
    def kronecker_delta(self, a, b):
        """Kronecker delta δ(a, b)."""
        return 1 if a == b else 0

    # -------------------------------------------------------------------------
    def F_n(self):
        """
        Implements Eq. (A14) from Kalinikos & Slavin (1986).
        Computes the form factor for mode n.
        """
        return (2 / (self.ksw * self.d)) * (1 - (-1) ** self.n * np.exp(-self.ksw * self.d))

    # -------------------------------------------------------------------------
    def P_nn_unpinned(self):
        """
        Implements Eq. (A12) from Kalinikos & Slavin (1986).
        Returns P_nn for totally unpinned surface spins (n = 0, 1, 2, ...).
        """
        k_ratio = self.ksw / self.ktot
        delta_term = 1 / (1 + self.kronecker_delta(0, self.n))
        return k_ratio ** 2 - (k_ratio ** 4) * self.F_n() * delta_term

    # -------------------------------------------------------------------------
    def P_nn_pinned(self):
        """
        Implements Eq. (A10) from Kalinikos & Slavin (1986).
        Returns P_nn for totally pinned surface spins (n = 1, 2, 3, ...).
        """
        k_ratio = self.ksw / self.ktot
        return k_ratio ** 2 + (self.ksw * self.kn / self.ktot ** 2) ** 2 * self.F_n()

    # -------------------------------------------------------------------------
    def F_nn(self):
        """
        Implements Eq. (46) from Kalinikos & Slavin (1986).
        Computes the dipole-exchange coupling correction term.
        """
        P_nn = self.P_nn_pinned() if self.pinning_cond else self.P_nn_unpinned()
        term1 = 1 - P_nn * (1 + np.cos(self.phi) ** 2)
        term2 = (P_nn * (1 - P_nn) * np.sin(self.phi) ** 2) / (
            self.OmgH + self.Lambdaex * self.ktot ** 2
        )
        return P_nn + np.sin(self.theta) ** 2 * (term1 + term2)

    # -------------------------------------------------------------------------
    def est_prop_sw_freq(self):
        """
        Implements Eq. (45) from Kalinikos & Slavin (1986).
        Returns the estimated propagating spin-wave frequency in Hz.
        """
        return self.fm * np.sqrt(
            (self.OmgH + self.Lambdaex * self.ktot ** 2)
            * (self.OmgH + self.Lambdaex * self.ktot ** 2 + self.F_nn())
        )
