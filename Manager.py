import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

#Klasse mit der ich alles managen kann
#LPT
from Aktienprognose import Aktienprognose





#startvariante, das ist meine hauptfunktion, damit wird Programm gestartet
if __name__ == "__main__":

    prognose1 = Aktienprognose
    print(prognose1)
    print("alles OK")