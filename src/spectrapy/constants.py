"""Physical constants (SI, exact values of the 2019 SI where applicable)."""

H = 6.62607015e-34          #: Planck constant [J s]
C = 299_792_458.0           #: speed of light in vacuum [m/s]
K_B = 1.380649e-23          #: Boltzmann constant [J/K]
SIGMA = 5.670374419e-8      #: Stefan-Boltzmann constant [W m^-2 K^-4]

C1L = 2.0 * H * C**2        #: first radiation constant for spectral radiance [W m^2 sr^-1]
C2 = H * C / K_B            #: second radiation constant [m K] (= 1.438776877e-2)
C2_CIE = 1.4388e-2          #: second radiation constant used by CIE 015 for Planckian radiators [m K]

K_M = 683.002               #: maximum luminous efficacy of photopic vision [lm/W]
HC_EV_NM = H * C / 1.602176634e-19 * 1e9  #: h c in eV nm (= 1239.84198...)
