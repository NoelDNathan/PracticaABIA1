from typing import List, Generator, Set
from abia_energia import *
from search import Problem, hill_climbing


# Our Methods

def distance(obj1, obj2):
    x = obj1.CoordX - obj2.CoordX
    y = obj1.CoordY - obj2.CoordY
    return (x**2 + y ** 2)**(1/2)

def electry_supplied_to_client(client: Cliente, power_plant: Central) -> float:
    dist = distance(client, power_plant)
    return client.Consumo * (1 + VEnergia.loss(dist))
    # return client.Consumo / (1 - VEnergia.loss(dist))



# Problem params

class ProblemParameters(object):
    def __init__(self, clients_vector: Clientes, power_plants_vector: Centrales) -> None:
        self.clients_vector = clients_vector
        self.power_plants_vector = power_plants_vector

    def __repr__(self):
        return f"clients_vector={self.clients_vector}\n\npower_plants_vector={self.power_plants_vector})"



# Operators

class Operator(object):
    pass


class MoveClient(Operator):
    def __init__(self, id_client, id_destination_PwP):
        self.id_client = id_client
        self.id_destination_PwP = id_destination_PwP

    def __repr__(self) -> str:
        return f"Client {self.id_client} has been moved to power plant {self.id_destination_PwP}"


class SwapClients(Operator):
    def __init__(self, id_client1, id_client2):
        self.id_client1 = id_client1
        self.id_client2 = id_client2

    def __repr__(self) -> str:
        return f"Swap between Client {self.id_client1} and Client {self.id_client2}"


class RemoveNGClient(Operator):
    def __init__(self, p_i: int, p_j: int):
        pass

    def __repr__(self) -> str:
        pass



# Generators initial states

# -> StateRepresentation:
def generate_simple_initial_state(params: ProblemParameters):
    remaining_energies = list()
    real_consumption = list()
    client_power_plant = list()

    ids_power_plants = list()

    for id_PwP, power_plant in enumerate(params.power_plants_vector):
        ids_power_plants.append(id_PwP)
        remaining_energies.append(power_plant.Produccion)
        row = list()

        for client in params.clients_vector:
            consumtion = electry_supplied_to_client(client, power_plant)
            row.append(consumtion)
        real_consumption.append(row)

    id_PwP = ids_power_plants.pop()

    for id_client, client in enumerate(params.clients_vector):
        if client.Contrato == NOGARANTIZADO:
            client_power_plant.append(-1)
            continue

        consum = real_consumption[id_PwP][id_client]
        while True:
            if consum < remaining_energies[id_PwP]:
                client_power_plant.append(id_PwP)
                remaining_energies[id_PwP] -= consum
                break

            id_PwP = ids_power_plants.pop()

    return StateRepresentation(params, client_power_plant, remaining_energies, real_consumption)


def generate_complex_initial_state():
    pass



# State Representation

class StateRepresentation(object):
    def __init__(self, params: ProblemParameters, client_power_plant: List[int], remaining_energies: List[float], real_consumption: List[List[float]]):
        self.params = params
        self.client_power_plant = client_power_plant
        self.remaining_energies = remaining_energies
        self.real_consumption = real_consumption

    def copy(self):
        return StateRepresentation(self.params, self.client_power_plant.copy(), self.remaining_energies.copy(), self.real_consumption.copy())

    def __repr__(self) -> str:
        return f"client_power_plant: {self.client_power_plant}"

    def generate_actions(self) -> Generator[Operator, None, None]:
        for id_client1 in range(len(self.params.clients_vector)):
            for id_PwP in range(len(self.params.power_plants_vector)):
                if self.client_power_plant[id_client1] == id_PwP:
                    continue

                csm_client = self.real_consumption[id_PwP][id_client1]

                if csm_client < self.remaining_energies[id_PwP]:
                    yield MoveClient(id_client1, id_PwP)

            for id_client2 in range(len(self.params.clients_vector)):
                if id_client1 == id_client2:
                    continue

                id_PwP1 = self.client_power_plant[id_client1]
                id_PwP2 = self.client_power_plant[id_client2]

                if id_PwP1 == id_PwP2:
                    continue

                if id_PwP1 == -1 or id_PwP2 == -1:
                    continue

                csm_pwp1_cli1 = self.real_consumption[id_PwP1][id_client1]
                csm_pwp2_cli1 = self.real_consumption[id_PwP2][id_client1]

                csm_pwp1_cli2 = self.real_consumption[id_PwP1][id_client2]
                csm_pwp2_cli2 = self.real_consumption[id_PwP2][id_client2]

                remain1 = self.remaining_energies[id_PwP1]
                remain2 = self.remaining_energies[id_PwP2]

                if (csm_pwp1_cli2 - csm_pwp1_cli1 < remain1 and csm_pwp2_cli1 - csm_pwp2_cli2 < remain2):
                    yield SwapClients(id_client1, id_client2)

    def apply_actions(self, action: Operator):  # -> StateRepresentation:
        new_state = self.copy()

        if isinstance(action, MoveClient):
            id_client = action.id_client
            id_PwP1 = self.client_power_plant[id_client]
            id_PwP2 = action.id_destination_PwP
            new_state.client_power_plant[id_client] = id_PwP2 = action.id_destination_PwP

            if id_PwP1 != -1:
                new_state.remaining_energies[id_PwP1] += self.real_consumption[id_PwP1][id_client]

            new_state.remaining_energies[id_PwP2] -= self.real_consumption[id_PwP2][id_client]

        elif isinstance(action, SwapClients):
            id_client1 = action.id_client1
            id_client2 = action.id_client2

            id_PwP1 = self.client_power_plant[id_client1]
            id_PwP2 = self.client_power_plant[id_client2]

            new_state.client_power_plant[id_client1] = id_PwP2
            new_state.client_power_plant[id_client2] = id_PwP1

            new_state.remaining_energies[id_PwP1] += self.real_consumption[id_PwP1][id_client1] - self.real_consumption[id_PwP1][id_client2]
            new_state.remaining_energies[id_PwP2] += self.real_consumption[id_PwP2][id_client2] - self.real_consumption[id_PwP2][id_client1]

        return new_state

    def heuristic(self) -> float:
        gain = 0

        for id_central, central in enumerate(self.params.power_plants_vector):
            if central.Produccion == self.remaining_energies[id_central]:
                gain -= VEnergia.stop_cost(central.Tipo)
            else:
                gain -= VEnergia.costs_production_mw(
                    central.Tipo) * central.Produccion + VEnergia.daily_cost(central.Tipo)

        for id_client, client in enumerate(self.params.clients_vector):
            if self.client_power_plant[id_client] == -1:
                gain -= VEnergia.tarifa_cliente_penalizacion(
                    client.Tipo) * client.Consumo
                continue

            if client.Contrato == 0:
                gain += VEnergia.tarifa_cliente_garantizada(
                    client.Tipo) * client.Consumo
            else:
                gain += VEnergia.tarifa_cliente_no_garantizada(
                    client.Tipo) * client.Consumo

        return gain



# Clase del problema

class EnergyProblem(Problem):
    def __init__(self, initial_state: StateRepresentation):
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operator, None, None]:
        return state.generate_actions()

    def result(self, state: StateRepresentation, action: Operator) -> StateRepresentation:
        return state.apply_actions(action)

    def value(self, state: StateRepresentation) -> float:
        return state.heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False



# Ejecución del programa.

clientes = Clientes(ncl=1000, propc=[0.4, 0.3, 0.3], propg=1, seed=44)
centrales = Centrales(centrales_por_tipo=[20, 10, 10], seed=44)
parametros = ProblemParameters(clients_vector=clientes, power_plants_vector=centrales)
estado_inicial = generate_simple_initial_state(params=parametros)

print(hill_climbing(EnergyProblem(estado_inicial)))
