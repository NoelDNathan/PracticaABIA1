from abia_energia import *

class ProblemParameters(object):
    def __init__(self, clients_vector: Clientes, power_plants_vector: Centrales) -> None:
        self.clients_vector = clients_vector
        self.power_plants_vector = power_plants_vector

    def __repr__(self):
        return f"clients_vector={self.clients_vector}\n\npower_plants_vector={self.power_plants_vector})"
