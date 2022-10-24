from typing import List, Generator, Set
from abia_energia import *
from search import Problem, hill_climbing
import math
import numpy as np
import time
from itertools import cycle

from utils import F


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
    def __init__(self, id_client: int):
        self.id_client = id_client

    def __repr__(self) -> str:
        return f"Client {self.id_client} has been removed"


# Generators initial states

# -> StateRepresentation:

def generate_state_vars(params: ProblemParameters):
    remaining_energies = list()
    real_consumption = list()

    for power_plant in params.power_plants_vector:
        remaining_energies.append(power_plant.Produccion)
        row = list()

        for client in params.clients_vector:
            consumtion = electry_supplied_to_client(client, power_plant)
            row.append(consumtion)
        real_consumption.append(row)

    remaining_energies = np.array(remaining_energies, dtype=float)
    real_consumption = np.array(real_consumption, dtype=float)
    return remaining_energies, real_consumption


def calculate_gains(params, remaining_energies, client_power_plant):
    gain = 0

    for id_central, central in enumerate(params.power_plants_vector):
        if central.Produccion == remaining_energies[id_central]:
            gain -= VEnergia.stop_cost(central.Tipo)
        else:
            gain -= VEnergia.costs_production_mw(central.Tipo) * \
                central.Produccion + VEnergia.daily_cost(central.Tipo)

    for id_client, client in enumerate(params.clients_vector):
        if client_power_plant[id_client] == -1:
            gain -= VEnergia.tarifa_cliente_penalizacion(client.Tipo) * client.Consumo
            continue

        if client.Contrato == 0:
            gain += VEnergia.tarifa_cliente_garantizada(client.Tipo) * client.Consumo
        else:
            gain += VEnergia.tarifa_cliente_no_garantizada(client.Tipo) * client.Consumo

    return gain


def generate_simple_initial_state(params: ProblemParameters):
    remaining_energies, real_consumption = generate_state_vars(params)
    # print(remaining_energies)
    client_power_plant = list()

    ids_power_plants = list(range(len(params.power_plants_vector)))

    id_PwP = ids_power_plants.pop()

    for id_client, client in enumerate(params.clients_vector):
        if client.Contrato == NOGARANTIZADO:
            client_power_plant.append(-1)
            continue

        consum = real_consumption[id_PwP][id_client]
        # print(consum)
        # print(remaining_energies[0])
        while True:
            if consum < remaining_energies[id_PwP]:
                client_power_plant.append(id_PwP)
                remaining_energies[id_PwP] -= consum
                break

            id_PwP = ids_power_plants.pop()

    client_power_plant = np.array(client_power_plant)

    gain = calculate_gains(params, remaining_energies, client_power_plant)

    return StateRepresentation(params, client_power_plant, remaining_energies, real_consumption, gain)


def generate_complex_initial_state(params: ProblemParameters):
    remaining_energies, real_consumption = generate_state_vars(params)


def generate_simple_initial_state2(params: ProblemParameters, worst=False):
    remaining_energies, real_consumption = generate_state_vars(params)
    client_power_plant = list()
    numClients = len(params.clients_vector)
    numPwP = len(params.power_plants_vector)
    id_client = 0
    id_PwP = 0
    cycle = 0
    MAX_NUM_CYCLE = 15  # Max num
    while id_client < numClients:
        if params.clients_vector[id_client].Tipo == NOGARANTIZADO:
            client_power_plant.append(-1)

        elif remaining_energies[id_PwP] > real_consumption[id_PwP][id_client]:
            remaining_energies[id_PwP] -= real_consumption[id_PwP][id_client]
            client_power_plant.append(id_PwP)
        else:
            # Si no se han cumplido ninguna de las dos condiciones, el cliente no es añadido a la central
            # Por eso pasamos a mirar la siguiente central
            id_PwP += 1

            # Si nos exedemos del número de centrales, volvemos a poner este valor a cero
            if id_PwP == numPwP:
                id_PwP = 0
                cycle += 1
                if cycle == MAX_NUM_CYCLE:
                    raise Exception("Sorry, max number of cycles did it")
            continue

        if worst:
            id_PwP += 1
            if id_PwP == numPwP:
                id_PwP = 0
                cycle += 1
                if cycle == MAX_NUM_CYCLE:
                    raise Exception("Sorry, max number of cycles did it")
        id_client += 1

    client_power_plant = np.array(client_power_plant)

    gain = calculate_gains(params, remaining_energies, client_power_plant)

    return StateRepresentation(params, client_power_plant, remaining_energies, real_consumption, gain)

# State Representation


class StateRepresentation(object):
    def __init__(self, params: ProblemParameters, client_power_plant: List[int], remaining_energies: List[float], real_consumption: List[List[float]], gain: float):
        self.params = params
        self.client_power_plant = client_power_plant
        self.remaining_energies = remaining_energies
        self.real_consumption = real_consumption
        self.gain = gain

    def copy(self):
        return StateRepresentation(self.params, self.client_power_plant.copy(), self.remaining_energies.copy(), self.real_consumption, self.gain)

    def __repr__(self) -> str:
        return f"client_power_plant: {self.client_power_plant}"

    def generate_actions(self) -> Generator[Operator, None, None]:
        numClients = len(self.params.clients_vector)
        numCentrals = len(self.params.power_plants_vector)

        for id_client1 in range(numClients):
            # MoveClient
            for id_PwP in range(numCentrals):
                if self.client_power_plant[id_client1] == id_PwP:
                    continue

                csm_client = self.real_consumption[id_PwP][id_client1]

                if csm_client < self.remaining_energies[id_PwP]:
                    yield MoveClient(id_client1, id_PwP)

            # SwapClients
            if id_client1 != numClients - 1:

                for id_client2 in range(id_client1 + 1, numClients):
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

                    if csm_pwp1_cli2 - csm_pwp1_cli1 < remain1 and csm_pwp2_cli1 - csm_pwp2_cli2 < remain2:
                        yield SwapClients(id_client1, id_client2)

            # Remove client
            # if self.params.clients_vector[id_client1].Tipo == NOGARANTIZADO:
            #     yield RemoveNGClient(id_client1)

    def apply_action(self, action: Operator):  # -> StateRepresentation:
        new_state = self.copy()

        if isinstance(action, MoveClient):
            id_client = action.id_client
            client = self.params.clients_vector[id_client]

            id_PwP1 = self.client_power_plant[id_client]
            PwP1 = self.params.power_plants_vector[id_PwP1]

            id_PwP2 = action.id_destination_PwP
            PwP2 = self.params.power_plants_vector[id_PwP2]

            new_state.client_power_plant[id_client] = id_PwP2

            if id_PwP1 != -1:
                new_state.remaining_energies[id_PwP1] += self.real_consumption[id_PwP1][id_client]

                # La central pasa a estar apagada (sumar el coste de tenerla encendida y restar el de que este apagada).
                # Solo puede pasar si el cliente estaba asignado a alguna central.
                if new_state.remaining_energies[id_PwP1] == PwP1.Produccion:
                    new_state.gain += VEnergia.costs_production_mw(PwP1.Tipo) * PwP1.Produccion + VEnergia.daily_cost(PwP1.Tipo) \
                        - VEnergia.stop_cost(PwP1.Tipo)

            # Si el cliente no estaba asignado a ninguna central, hay que tener en cuenta su coste.
            else:
                new_state.gain += VEnergia.tarifa_cliente_penalizacion(client.Tipo) * client.Consumo

            # Si la central a la que se mueve estaba apagada (sumar el coste de tenerla apagada y restar el de tenerla encendida).
            if new_state.remaining_energies[id_PwP2] == PwP2.Produccion:
                new_state.gain -= VEnergia.costs_production_mw(PwP2.Tipo) * PwP2.Produccion + VEnergia.daily_cost(PwP2.Tipo) \
                    + VEnergia.stop_cost(PwP2.Tipo)

            new_state.remaining_energies[id_PwP2] -= self.real_consumption[id_PwP2][id_client]

        elif isinstance(action, SwapClients):
            id_client1 = action.id_client1
            id_client2 = action.id_client2

            id_PwP1 = self.client_power_plant[id_client1]
            id_PwP2 = self.client_power_plant[id_client2]

            new_state.client_power_plant[id_client1] = id_PwP2
            new_state.client_power_plant[id_client2] = id_PwP1

            new_state.remaining_energies[id_PwP1] += self.real_consumption[id_PwP1][id_client1] - \
                self.real_consumption[id_PwP1][id_client2]
            new_state.remaining_energies[id_PwP2] += self.real_consumption[id_PwP2][id_client2] - \
                self.real_consumption[id_PwP2][id_client1]

        # elif isinstance(action, RemoveNGClient):
        #     id_client = action.id_client

        #     new_state.client_power_plant[id_client] = -1

        #     id_PwP = self.client_power_plant[id_client]

        #     new_state.remaining_energies[id_PwP] += self.real_consumption[id_PwP][id_client]

        return new_state

    def apply_action_without_copy(self, action: Operator):

        if isinstance(action, MoveClient):
            id_client = action.id_client
            client = self.params.clients_vector[id_client]

            id_PwP1 = self.client_power_plant[id_client]
            PwP1 = self.params.power_plants_vector[id_PwP1]

            id_PwP2 = action.id_destination_PwP
            PwP2 = self.params.power_plants_vector[id_PwP2]

            self.client_power_plant[id_client] = id_PwP2

            if id_PwP1 != -1:
                self.remaining_energies[id_PwP1] += self.real_consumption[id_PwP1][id_client]

                # La central pasa a estar apagada (sumar el coste de tenerla encendida y restar el de que este apagada).
                # Solo puede pasar si el cliente estaba asignado a alguna central.
                if self.remaining_energies[id_PwP1] == PwP1.Produccion:
                    self.gain += VEnergia.costs_production_mw(PwP1.Tipo) * PwP1.Produccion + VEnergia.daily_cost(PwP1.Tipo) \
                        - VEnergia.stop_cost(PwP1.Tipo)

            # Si el cliente no estaba asignado a ninguna central, hay que tener en cuenta su coste.
            else:
                if client.Contrato == 0:
                    self.gain += VEnergia.tarifa_cliente_garantizada(
                        client.Tipo) * client.Consumo
                else:
                    self.gain += VEnergia.tarifa_cliente_no_garantizada(
                        client.Tipo) * client.Consumo

            # Si la central a la que se mueve estaba apagada (sumar el coste de tenerla apagada y restar el de tenerla encendida).
            if self.remaining_energies[id_PwP2] == PwP2.Produccion:
                self.gain -= VEnergia.costs_production_mw(PwP2.Tipo) * PwP2.Produccion + VEnergia.daily_cost(PwP2.Tipo) \
                    + VEnergia.stop_cost(PwP2.Tipo)

            self.remaining_energies[id_PwP2] -= self.real_consumption[id_PwP2][id_client]

        elif isinstance(action, SwapClients):
            id_client1 = action.id_client1
            id_client2 = action.id_client2

            id_PwP1 = self.client_power_plant[id_client1]
            id_PwP2 = self.client_power_plant[id_client2]

            self.client_power_plant[id_client1] = id_PwP2
            self.client_power_plant[id_client2] = id_PwP1

            self.remaining_energies[id_PwP1] += self.real_consumption[id_PwP1][id_client1] - \
                self.real_consumption[id_PwP1][id_client2]
            self.remaining_energies[id_PwP2] += self.real_consumption[id_PwP2][id_client2] - \
                self.real_consumption[id_PwP2][id_client1]

        return self

    def better_heuristic(self) -> float:
        return self.gain

    def bad_heuristic(self) -> float:
        return calculate_gains(self.params, self.remaining_energies, self.client_power_plant)

    def heuristic_simple_entropy(self) -> float:
        entropy = 0
        for id_central, power_plant in enumerate(self.params.power_plants_vector):
            remain = self.remaining_energies[id_central]
            max_prod = power_plant.Produccion

            occupancy = 1 - (max_prod - remain / max_prod)
            entropy -= occupancy * math.log(occupancy)
        return entropy


# Clase del problema


class EnergyProblem(Problem):
    def __init__(self, initial_state: StateRepresentation):
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operator, None, None]:
        return state.generate_actions()

    def result(self, state: StateRepresentation, action: Operator) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        return state.better_heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False


# Ejecución del programa.
# clientes = Clientes(ncl=100, propc=[0.4, 0.3, 0.3], propg=1, seed=44)
# centrales = Centrales(centrales_por_tipo=[20, 10, 10], seed=44)
# parametros = ProblemParameters(clients_vector=clientes, power_plants_vector=centrales)
# estado_inicial = generate_simple_initial_state2(params=parametros, worst=False)

clientes = Clientes(ncl=1000, propc=[0.25, 0.3, 0.45], propg=0.75, seed=1234)
centrales = Centrales(centrales_por_tipo=[5, 10, 25], seed=1234)
parametros = ProblemParameters(clients_vector=clientes, power_plants_vector=centrales)
estado_inicial = generate_simple_initial_state(params=parametros)

print(estado_inicial)
print(estado_inicial.better_heuristic())
print()

n = estado_inicial.client_power_plant
a = 0
for x in n:
    if x == -1:
        a += 1
print(a)

start = time.time()
ejecucion = hill_climbing(EnergyProblem(estado_inicial))
print(time.time() - start)
print(ejecucion)
print(ejecucion.better_heuristic())

m = ejecucion.client_power_plant
b = 0
for x in m:
    if x == -1:
        b += 1
print(b)