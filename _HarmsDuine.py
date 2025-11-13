import numpy as np

class DispersionModel():

    def __init__(self, film_prop, **kwargs):
        #universal constants
        self.gammaLL = 1.76e11
        self.mu0 = 4e-07 * np.pi

        # film properties
        self.Ms = film_prop['Ms']
        # spin stifness (Wb) not exchange stifness (J/m)
        self.Aex = film_prop['Aex']
        self.lex = np.sqrt(2 * self.Aex / (self.mu0 * self.Ms**2)) # refer to sec.2.1, pg.2 
        self.d = film_prop['d'] / self.lex

        # wave number information
        self.n = kwargs.get('mode_no', 0)
        self.kn = (self.n * np.pi) / self.d                 # refer to sec.3.1, pg.3
        self.ksw = kwargs.get('ksw', [0.0]) * self.lex
        if self.ksw.min() == 0:
            raise ValueError("ksw (in-plane wave number) cannot be zero.")

        # external field normalized by satuaration magnetization
        self.OmgH = kwargs.get('Heff', 0.0) / self.Ms                # refer to sec.2.1, pg.2
        if self.OmgH == 0:
            raise ValueError("Ratio between effective field and saturation magnetization cannot be zero.")
        self.fm = (self.gammaLL * self.mu0 * self.Ms) / (2 * np.pi)

        # mode number dependent constants
        self.alpha_n = 4 / (3 * np.pi**2 * (1 + self.n)**2)  # refer to Eq. 29
        self.a_nk = -1                                       # refer to Eq. 31a

    # -------------------------------------------------------------------------
    def kronecker_delta(self, a, b):
        """Kronecker delta δ(a, b)."""
        return 1 if a == b else 0

    # -------------------------------------------------------------------------
    # refer to sec.3.1.2, pg.4
    def gamma_nk(self):
        return 2 * (self.OmgH + 0.5 + self.ksw*2 + self.kn**2)

    # -------------------------------------------------------------------------
    # refer to Eq. 32a
    def B_nk(self):
        term1 = (1 - np.exp(-2 * self.ksw * self.d)) / (4 * self.gamma_nk() / self.d**2)
        term2 = (self.ksw * self.d)**2 + (self.n * np.pi)**2
        term3 = (self.ksw * self.d)**2 + (self.n * np.pi)**2 + self.kronecker_delta(self.n, 0) * np.pi**2 / 4
        term4 = 3 * (self.ksw * self.d)**2 + (self.n * np.pi)**2 + self.kronecker_delta(self.n, 0) * np.pi**2 / 4
        return term1 - term2 * term3 / term4

    # -------------------------------------------------------------------------
    # refer to Eq. 32b
    def C_nk(self):
        term1 = self.ksw**3 * self.d**5 / (2 * self.gamma_nk())
        term2 = 3 * (self.ksw * self.d)**2 + (self.n * np.pi)**2 + self.kronecker_delta(self.n, 0) * np.pi**2 / 4
        return term1 / term2

    # -------------------------------------------------------------------------
    # refer to Eq. 31b 
    def b_nk(self):
        return self.B_nk() - self.alpha_n * self.C_nk() - self.a_nk * (2 * self.n + 1) * np.pi**2

    # -------------------------------------------------------------------------
    # refer to Eq. 31c
    def c_nk(self):
        term1 = (4 - self.kronecker_delta(self.n, 0)) * self.C_nk()
        term2 = (self.B_nk() - self.alpha_n * self.C_nk()) * (2 * self.n + 1) * np.pi**2
        return term1 - term2

    # -------------------------------------------------------------------------
    # refer to Eq. 31d
    def d_nk(self):
        return -self.C_nk() * (2 - self.kronecker_delta(self.n, 0)) * (2 * self.n + 1) * np.pi**2

    # -------------------------------------------------------------------------
    # refer to Eq. 34a
    def Q_nk(self):
        term1 = self.d_nk() / self.a_nk
        term2 = (self.b_nk() * self.c_nk() / self.a_nk**2) * (1/3)
        term3 = (self.b_nk() / self.a_nk)**3 * (2/27) 
        return term1 - term2 + term3

    # -------------------------------------------------------------------------
    # refer to Eq. 34b
    def P_nk(self):
        term1 = self.c_nk() / self.a_nk
        term2 = (self.b_nk() / self.a_nk)**2 * (1/3)
        return term1 - term2

    # -------------------------------------------------------------------------
    # refer to Eq. 33a
    def z_nk(self):
        term1 = (self.b_nk() / self.a_nk) * (-1 / 3)
        term2 = np.sqrt(-4 * self.P_nk() / 3)
        term3 = np.cos((1 / 3) * np.arccos(1.5 * np.sqrt(-3 / self.P_nk()) * (self.Q_nk() / self.P_nk())) - (2 * np.pi /3))
        return term1 + term2 * term3

    # -------------------------------------------------------------------------
    # refer to Eq. 35
    def est_prop_sw_freq(self):
        return self.fm * np.sqrt((self.OmgH + 0.5 + self.ksw**2 + self.kn**2 + self.z_nk() / self.d**2)**2 - 0.25)