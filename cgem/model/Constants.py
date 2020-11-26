from ClassicSIR import ClassicSIR
from DHSIRModel import DHSIRModel

MOBILITY_TYPE_SYN = ['No', 'BA', 'ER']
NETWORK_TYPES = ["Complete", "BA", "ER", "base"]
SIR_COMPARTMENTS = ["Susceptible", "Infected", "Recovered"]
SEIR_COMPARTMENTS = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
SEIR_STATES = ['S', 'E', 'I_S', 'I_A', 'R']
# https://www.quora.com/What-is-the-average-amount-of-passengers-on-a-plane
AVG_DOMESTIC_FLIGHT_CAP = 90.7
AVG_INTERNATIONAL_FLIGHT_CAP = 141.5
AVG_FLIGHT_CAP = (AVG_DOMESTIC_FLIGHT_CAP + AVG_INTERNATIONAL_FLIGHT_CAP) / 2
COVID_MODELS = {'ClassicSIR': ClassicSIR, 'DHSIRModel': DHSIRModel, 'True': None}
COVID_MODEL_NAMES = ['ClassicSIR', 'DHSIRModel', 'True']
EXP_MODE = ['CAD_INT', 'CAD_WORLD', 'PROV_OTHER']
