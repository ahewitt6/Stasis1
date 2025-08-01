import PRyM.PRyM_thermo as PRyMthermo
import PRyM.PRyM_init as PRyMini
import numpy as np

enabled = False
T_start = PRyMini.stasis_params["T_start_K"] / PRyMini.MeV_to_Kelvin
T_end = PRyMini.stasis_params["T_end_K"] / PRyMini.MeV_to_Kelvin
frac_gamma         = 1.0 / 3.0      # branching fractions (sum to 1)
frac_e             = 1.0 / 3.0
frac_nu            = 1.0 / 3.0

MeV_to_Hz = 1.519267447e21          #  1/ħ  in  s⁻¹ MeV⁻¹

rho_start = None
fractions = {}
equations_of_state = {}
exponents = {}
branching = {}
Gamma = 0.0


def configure(params):
    """
    Initialize stasis parameters from the given dictionary.

    Expects keys:
    - 'enabled'
    - 'stasis_start_MeV'
    - 'stasis_end_MeV'
    - 'rho_m0'            (matter density at entry)
    - 'rho_rad_entry'     (radiation density at entry)
    """
    global enabled, T_start, T_end, rho_start, fractions, equations_of_state, exponents, branching, Gamma

    # enable flag (must exist)
    enabled = params['enabled']
    if not enabled:
        return

    # define stasis window edges (MeV)
    T_start = params['stasis_start_MeV']
    T_end   = params['stasis_end_MeV']

    # # entry densities
    # matter_start = params['rho_m0']
    # rad_start    = params['rho_rad_entry']
    # rho_start    = matter_start + rad_start

    # # compute base fractions
    # f_matter = matter_start / rho_start
    # f_rad    = rad_start    / rho_start
    fractions.clear()

    # Decide f_matter and f_rad from Ω_M (if given) or fallback to absolute densities
    # if 'Omega_M' in params:
    # user told us the matter fraction explicitly
    f_matter = float(params['Omega_M'])
    f_rad    = 1.0 - f_matter
    # reconstruct total density at entry so ρ_rad_entry = f_rad · ρ_start
    rho_rad_entry = float(params['rho_rad_entry'])
    rho_start     = rho_rad_entry / f_rad
    # else:
    #     # fallback: infer from absolute densities
    #     matter_start = float(params['rho_m0'])
    #     rad_start    = float(params['rho_rad_entry'])
    #     rho_start    = matter_start + rad_start
    #     f_matter     = matter_start / rho_start
    #     f_rad        = rad_start    / rho_start

    # distribute radiation fraction
    # if 'radiation_distribution' in params:
    rd = params['radiation_distribution']
    # ensure keys exist
    b_gamma = rd['gamma']; b_e = rd['e']; b_nu = rd['nu']
    # total_w = w_gamma + w_e + w_nu
    fractions['gamma'] = f_rad * b_gamma 
    fractions['e']     = f_rad * b_e 
    fractions['nu']    = f_rad * b_nu 

    # fractions['gamma'] = f_matter * b_gamma 
    # fractions['e']     = f_matter * b_e 
    # fractions['nu']    = f_matter * b_nu

    fractions['gamma_after'] = f_matter * b_gamma 
    fractions['e_after']     = f_matter * b_e 
    fractions['nu_after']    = f_matter * b_nu 

    # else:
    #     # fallback: use actual densities at T_start
    #     rho_g_star   = PRyMthermo.rho_g(T_start)
    #     rho_e_star   = PRyMthermo.rho_e(T_start)
    #     rho_nu_star  = 3 * PRyMthermo.rho_nu(T_start)
    #     fractions['gamma'] = rho_g_star  / rho_start
    #     fractions['e']     = rho_e_star  / rho_start
    #     fractions['nu']    = rho_nu_star / rho_start

    # matter fraction
    fractions['dm'] = f_matter
    # fractions['dm'] = 0

    # set equations of state
    equations_of_state = {
        'gamma': 1/3,
        'e':      1/3,
        'nu':     1/3,
        'dm':     0.0,
        'gamma_after': 1/3,
        'e_after':      1/3,
        'nu_after':     1/3,
    }

    branching = {
    "gamma": 0.6,
    "e":     0.3,
    "nu":    0.1,
    "dm": f_matter
    }

    # compute effective exponent
    w_eff = sum(equations_of_state[s] * fractions[s] for s in fractions)
    exp_tot = 3.0 * (1.0 + w_eff)
    exponents = {s: exp_tot for s in fractions}
    # if params.get("dm_constant_in_stasis", False):
    #     exponents["dm"] = 0.0

    Gamma = params['kappa'] * 1.66 * 2. * T_start**2 / PRyMini.Mpl

# Smooth progress: 0 at T_start  → 1 at T_end
def progress(T):
    if T >= T_start:
        return 0.0
    if T <= T_end:
        return 1.0
    x = (T_start - T) / (T_start - T_end)   # 0 → 1 inside the band
    return 3.0*x*x - 2.0*x*x*x              # C¹ sigmoid

def in_stasis_window(T):
    """Check if temperature T (MeV) lies inside the stasis window."""
    return enabled and (T_end <= T <= T_start)


def rho_species(T, species):
    """Energy density of a given species during stasis."""
    # return fractions[species] * rho_start * (T / T_start) ** exponents[species]
    return fractions[species]* rho_start * (T / T_start) ** exponents[species]

# def rho_species(T, species):
#     """Energy density of a given species during stasis."""

#     # baseline   = what the SM would give that species at T_start
#     baseline = fractions[species] * rho_start * (T / T_start) ** exponents[species]

#     if species == "dm":
#         # tower energy still present = (1 - progress) × initial
#         # return (1.0 - progress(T)) * baseline
#         return baseline

#     # everything the tower has **already** shed goes to radiation channels
#     injected = progress(T) * fractions['dm'] * rho_start * (T / T_start) ** 4

#     return baseline + branching[species] * injected


def drho_species_dT(T, species):
    """Temperature derivative of energy density during stasis."""
    exp_i = exponents[species]
    return exp_i * rho_species(T, species) / T


def pressure_species(T, species):
    """Pressure for a given species during stasis."""
    w_i = equations_of_state[species]
    return w_i * rho_species(T, species)


def Qdot_total(T):
    if not enabled:
        return 0.0
    return Gamma * rho_species(T,'dm') * MeV_to_Hz         # Γ ρ_m


def Qdot_plasma(Tg_MeV):
    """Power dumped into the tightly coupled γ + e± bath."""
    return (fractions['gamma'] + fractions['e']) * Qdot_total(Tg_MeV)

def Qdot_nu(Tnu_MeV):
    """Power dumped into *all* neutrinos (sum over flavours)."""
    return fractions['nu'] * Qdot_total(Tnu_MeV)
