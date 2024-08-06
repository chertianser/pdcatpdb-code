import numpy as np
from math import exp, sqrt, pi, log
import sys
import logging

################################################################################

#: The Avogadro constant :math:`N_\mathrm{A}` in :math:`\mathrm{mol^{-1}}`       
Na = 6.02214179e23

#: The gas law constant :math:`R` in :math:`\mathrm{J/mol \cdot K}`
R = 8.314472

#: :math:`\pi = 3.14159 \ldots`      
pi = float(pi)

#: The speed of light in a vacuum :math:`c` in :math:`\mathrm{m/s}`
c = 299792458

#: The Planck constant :math:`h` in :math:`\mathrm{J \cdot s}`
h = 6.62606896e-34

#: The Boltzmann constant :math:`k_\mathrm{B}` in :math:`\mathrm{J/K}`
kB = 1.3806504e-23

class Wigner():
    """
    A tunneling model based on the Wigner formula. The attributes are:

    =============== =============================================================
    Attribute       Description
    =============== =============================================================
    `frequency`     The imaginary frequency of the transition state
    =============== =============================================================
    
    """

    def __init__(self, frequency):
        self.frequency = float(frequency) # cm-1


    def calculate_tunneling_factor(self, T):
        """
        Calculate and return the value of the Wigner tunneling correction for
        the reaction at the temperature `T` in K.
        """
        frequency = abs(self.frequency) * c * 100.0 # Is already SI units conversion
        factor = h * frequency / (kB * float(T))
        return 1.0 + factor * factor / 24.0
                
################################################################################

class Eckart():
    """
    A tunneling model based on the Eckart model. The attributes are:

    =============== =============================================================
    Attribute       Description
    =============== =============================================================
    `frequency`     The imaginary frequency of the transition state
    `E0_reac`       The ground-state energy of the reactants
    `E0_TS`         The ground-state energy of the transition state
    `E0_prod`       The ground-state energy of the products
    =============== =============================================================
    
    If `E0_prod` is not given, it is assumed to be the same as the reactants;
    this results in the so-called "symmetric" Eckart model. Providing 
    `E0_prod`, and thereby using the "asymmetric" Eckart model, is the
    recommended approach.
    """

    def __init__(self, frequency, E0_reac, E0_TS, E0_prod=None):
        self.frequency = float(frequency) # cm-1
        self.E0_reac = float(E0_reac)*1000*2625.5 # hartrees -> J/mol
        self.E0_TS = float(E0_TS)*1000*2625.5 # hartrees -> J/mol
        self.E0_prod = float(E0_prod)*1000*2625.5 # hartrees -> J/mol

    def calculate_tunneling_factor(self, T):
        """
        Calculate and return the value of the Eckart tunneling correction for
        the reaction at the temperature `T` in K.
        """
        #cdef double E0_reac, E0_prod, E0_TS
        #cdef double E0, dE, beta, dV1, dV2
        #cdef np.ndarray Elist, kappaE
                
        T = float(T) # laziness

        beta = 1. / (R * T)  # [=] mol/J

        E0_reac = self.E0_reac
        E0_TS = self.E0_TS
        E0_prod = self.E0_prod

        # Calculate intermediate constants
        if E0_reac > E0_prod:
            E0 = E0_reac
            dV1 = E0_TS - E0_reac
            dV2 = E0_TS - E0_prod
        else:
            E0 = E0_prod
            dV1 = E0_TS - E0_prod
            dV2 = E0_TS - E0_reac

        if dV1 < 0 or dV2 < 0:
            logging.info('\n')
            logging.error('Got the following wells:\nReactants: {0:g} kJ/mol\nTS: {1:g} kJ/mol\n'
                          'Products: {2:g} kJ/mol\n'.format(E0_reac / 1000., E0_TS / 1000., E0_prod / 1000.))
            raise ValueError('One or both of the barrier heights of {0:g} and {1:g} kJ/mol encountered in Eckart '
                             'method are invalid.'.format(dV1 / 1000., dV2 / 1000.))

        # Ensure that dV1 is smaller than dV2
        assert dV1 <= dV2

        # Evaluate microcanonical tunneling function kappa(E)
        dE = 100.
        Elist = np.arange(E0, E0 + 2. * (E0_TS - E0) + 40. * R * T, dE)
        kappaE = self.calculate_tunneling_function(Elist)

        # Integrate to get kappa(T)
        kappa = exp(dV1 * beta) * np.sum(kappaE * np.exp(-beta * (Elist - E0))) * dE * beta

        # Return the calculated Eckart correction
        return kappa

    def calculate_tunneling_function(self, Elist):
        """
        Calculate and return the value of the Eckart tunneling function for
        the reaction at the energies `e_list` in J/mol.
        """
        
        '''
                cdef double frequency, E0_reac, E0_prod, E0_TS
        cdef np.ndarray[np.float64_t, ndim=1] kappa, _Elist
        cdef double E0, dV1, dV2, alpha1, alpha2, E, xi, twopia, twopib, twopid
        cdef int r, r0
        '''

        frequency = abs(self.frequency) * h * c * 100. * Na # Is already SI units conversion
        E0_reac = self.E0_reac
        E0_TS = self.E0_TS
        E0_prod = self.E0_prod

        _Elist = Elist

        # Calculate intermediate constants
        if E0_reac > E0_prod:
            E0 = E0_reac
            dV1 = E0_TS - E0_reac
            dV2 = E0_TS - E0_prod
        else:
            E0 = E0_prod
            dV1 = E0_TS - E0_prod
            dV2 = E0_TS - E0_reac

        # Ensure that dV1 is smaller than dV2
        assert dV1 <= dV2

        alpha1 = 2 * pi * dV1 / frequency
        alpha2 = 2 * pi * dV2 / frequency

        kappa = np.zeros_like(Elist)
        for r0 in range(_Elist.shape[0]):
            if _Elist[r0] >= E0:
                break

        for r in range(r0, _Elist.shape[0]):
            E = _Elist[r]

            xi = (E - E0) / dV1
            # 2 * pi * a
            twopia = 2.*sqrt(alpha1*xi)/(1./sqrt(alpha1)+1./sqrt(alpha2))
            # 2 * pi * b
            twopib = 2.*sqrt(abs((xi-1.)*alpha1+alpha2))/(1./sqrt(alpha1)+1/sqrt(alpha2))
            # 2 * pi * d
            twopid = 2.*sqrt(abs(alpha1*alpha2-4*pi*pi/16.))

            # We use different approximate versions of the integrand to avoid
            # domain errors when evaluating cosh(x) for large x
            # If all of 2*pi*a, 2*pi*b, and 2*pi*d are sufficiently small,
            # compute as normal
            if twopia < 200. and twopib < 200. and twopid < 200.:
                kappa[r] = 1 - (np.cosh(twopia-twopib)+np.cosh(twopid)) / (np.cosh(twopia+twopib)+np.cosh(twopid))
            # If one of the following is true, then we can eliminate most of the
            # exponential terms after writing out the definition of cosh and
            # dividing all terms by exp(2*pi*d)
            elif twopia-twopib-twopid > 10 or twopib-twopia-twopid > 10 or twopia+twopib-twopid > 10:
                kappa[r] = 1 - exp(-2*twopia) - exp(-2*twopib) - exp(-twopia-twopib+twopid) - exp(-twopia-twopib-twopid)
            # Otherwise expand each cosh(x) in terms of its exponentials and divide
            # all terms by exp(2*pi*d) before evaluating
            else:
                kappa[r] = 1 - (exp(twopia-twopib-twopid) + exp(-twopia+twopib-twopid) + 1 + exp(-2*twopid)) / (exp(twopia+twopib-twopid) + exp(-twopia-twopib-twopid) + 1 + exp(-2*twopid))

        return kappa


if __name__ == "__main__":
        frequency = float(sys.argv[1])
        E0_reac = float(sys.argv[2])
        E0_TS = float(sys.argv[3])
        E0_prod = float(sys.argv[4])
        T = float(sys.argv[5])
        wigner_kappa = Wigner(frequency=frequency).calculate_tunneling_factor(T=T)
        eckart_kappa = Eckart(frequency=frequency, E0_reac=E0_reac, E0_TS=E0_TS, E0_prod=E0_prod).calculate_tunneling_factor(T=T)
        uncorr_barrier = (E0_TS-E0_reac)*2625.5*1000
        wigner_corr_barrier = uncorr_barrier - R*T*log(wigner_kappa)
        eckart_corr_barrier = uncorr_barrier - R*T*log(eckart_kappa)
        print(f"Wigner Kappa is {wigner_kappa}")
        print(f"Eckart Kappa is {eckart_kappa}")
        print(f"Uncorrected barrier is {uncorr_barrier/4184}")
        print(f"Wigner Corrected barrier is {wigner_corr_barrier/4184}")
        print(f"Eckart Corrected barrier is {eckart_corr_barrier/4184}")
