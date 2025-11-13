import numpy as np

class DispersionModel():

    """
        Description:
            This module implements the Prabhakar-Stancil dispersion model
            for ferromagnetic thin films, as described in:
            
            Prabhakar, A., and Stancil, D., D. 
            "Spin Waves: Theory and Applications." 
            Springer New York, NY.
            DOI: 10.1007/978-0-387-77865-5

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
                                                'MSSW', 'BVSW', 'FVSW'
                }

        Outputs:
            est_prop_sw_freq() : Returns the estimated propagating spin-wave frequency in Hz.
            est_sw_reso_freq() : Returns the estimated spin-wave resonance frequency in Hz.
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
        self.OmgH_ex = self.OmgH + self.Lambdaex * self.ktot**2
        self.fm = (self.gammaLL * self.mu0 * self.Ms) / (2 * np.pi)  # frequency scaling factor (Hz)

        # --- Geometry configuration ---
        self.config = kwargs.get('config', 'MSSW')

    # -------------------------------------------------------------------------    
    def est_prop_sw_freq(self):
        """
        Returns the estimated propagating spin-wave frequency in Hz.
        """
        if isinstance(self.config, str):
            if self.config == 'MSSW':
                term = (1 - np.exp(-2 * self.ksw * self.d)) / 4
                return self.fm * np.sqrt(self.OmgH_ex * (self.OmgH_ex + 1) + term)

            elif self.config == 'BVSW':
                trem = (1 - np.exp(-self.ksw * self.d)) / (self.ksw * self.d)
                return self.fm * np.sqrt(self.OmgH_ex * (self.OmgH_ex + term))

            elif self.config == 'FVSW':
                trem =  1 - ((1 - np.exp(-self.ksw * self.d)) / (self.ksw * self.d))
                return self.fm * np.sqrt(self.OmgH_ex * (self.OmgH_ex + term))
            
            else:
                raise ValueError("config must be one of {'MSSW', 'BVSW', 'FVSW'}.")
        else:
            raise ValueError("Invalid config format.")

    # -------------------------------------------------------------------------
    def est_sw_reso_freq(self):
        """
        Returns the estimated spin-wave resonance frequency in Hz.
        """
        OmgH_ex_1 = self.OmgH + self.Lambdaex * self.kn**2

        if isinstance(self.config, str):
            if self.config == 'Normal':
                return self.fm * OmgH_ex_1

            elif self.config == 'Tangential':
                return self.fm * np.sqrt(OmgH_ex_1 * (OmgH_ex_1 + 1))
            
            else:
                raise ValueError("config must be one of {'Normal', 'Tangential'}.")
        else:
            raise ValueError("Invalid config format.")
