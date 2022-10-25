import timeit
from http import client
from tkinter import N
from typing import List, Generator, Set
from abia_energia import *
from search import Problem, hill_climbing
import math
import numpy as np
import time

from utils import F

# Our Constants
NONPOWERPLANT = -1

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


class MergePwPs(Operator):
    def __init__(self, smaller, bigger, consumed) -> None:
        self.id_smaller = smaller
        self.id_bigger = bigger
        self.consumed = consumed

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
        if client_power_plant[id_client] == NONPOWERPLANT:
            gain -= VEnergia.tarifa_cliente_penalizacion(
                client.Tipo) * client.Consumo
            continue

        if client.Contrato == GARANTIZADO:
            gain += VEnergia.tarifa_cliente_garantizada(
                client.Tipo) * client.Consumo
        else:
            gain += VEnergia.tarifa_cliente_no_garantizada(
                client.Tipo) * client.Consumo

    return gain


def calculate_entropy(params: ProblemParameters, remaining_energies):
    total_entropy = 0
    for id, remain in enumerate(remaining_energies):
        max_prod = params.power_plants_vector[id].Produccion
        occupancy = 1 - (remain/max_prod)
        if occupancy > 0:
            total_entropy = total_entropy - (occupancy * math.log(occupancy))
    return -total_entropy


def generate_simple_initial_state(params: ProblemParameters):
    remaining_energies, real_consumption = generate_state_vars(params)
    client_power_plant = list()

    ids_power_plants = list(range(len(params.power_plants_vector)))

    id_PwP = ids_power_plants.pop()

    for id_client, client in enumerate(params.clients_vector):
        if client.Contrato == NOGARANTIZADO:
            client_power_plant.append(NONPOWERPLANT)
            continue

        consum = real_consumption[id_PwP][id_client]
        print(consum)
        print(remaining_energies[0])
        while True:
            if consum < remaining_energies[id_PwP]:
                client_power_plant.append(id_PwP)
                remaining_energies[id_PwP] -= consum
                break

            id_PwP = ids_power_plants.pop()

    client_power_plant = np.array(client_power_plant)

    gain = calculate_gains(params, remaining_energies, client_power_plant)

    return StateRepresentation(params, client_power_plant, remaining_energies, real_consumption, gain)


def generate_complex_initial_state(params: ProblemParameters, n: int, useNG: bool):
    remaining_energies, real_consumption = generate_state_vars(params)

    client_power_plant = [-1] * len(params.clients_vector)
    div = 100 // n

    zones = [[[] for i in range(n)]for _ in range(n)]

    def assign_power_plant(client_vector, isGuaranteed):
        for id, client in client_vector:
            X = min(client.CoordX // div, n - 1)
            Y = min(client.CoordY // div, n - 1)

            Z1 = 0
            Z2 = 0
            Z3 = 0
            dist = 0
            next_moveX = []
            next_moveY = []

            while True:
                if len(zones[Y + Z1][X + Z2]) > 0:
                    PwP = zones[Y + Z1][X + Z2][Z3]
                    consum = real_consumption[PwP][id]
                    if consum < remaining_energies[PwP]:
                        client_power_plant[id] = PwP
                        remaining_energies[PwP] -= consum
                        break
                    # Se van mirando las centrales de cada zona hasta que no quedan más
                    elif Z3 < len(zones[Y + Z1][X + Z2]) - 1:
                        Z3 += 1
                        continue

                if len(next_moveX) == 0:

                    dist += 1

                    # All region watched
                    if dist >= n:
                        if isGuaranteed:
                            raise Exception(
                                "Too much client than power plants")
                        else:
                            break

                    lim_left = -dist if X - dist >= 0 else 0
                    lim_down = -dist if Y - dist >= 0 else 0
                    lim_right = dist if X + dist < n else (n - 1) - X
                    lim_up = dist if Y + dist < n else (n - 1) - Y

                    # We will use pop, so the order is inversed that we are doing here

                    moves_axis_X = list(range(lim_left, lim_right + 1))
                    moves_axis_Y = list(range(lim_down, lim_up + 1))

                    # Go right
                    next_moveX += moves_axis_X
                    next_moveY += [lim_down] * len(moves_axis_X)

                    # Go up
                    next_moveX += [lim_right] * len(moves_axis_Y)
                    next_moveY += moves_axis_Y

                    # Go left
                    next_moveX += reversed(moves_axis_X)
                    next_moveY += [lim_up] * len(moves_axis_X)

                    # Go down
                    next_moveX += [lim_left] * len(moves_axis_Y)
                    next_moveY += reversed(moves_axis_Y)

                else:
                    Z3 = 0
                    Z1 = next_moveY.pop()
                    Z2 = next_moveX.pop()

    for id, PwP in enumerate(params.power_plants_vector):
        # podría passar que en algún caso muy extremo CoordX = 100 or CoordY
        # En ese caso queremos que su X or Y sea 9 y no 10
        X = min(PwP.CoordX // div, n - 1)
        Y = min(PwP.CoordY // div, n - 1)
        zones[Y][X].append(id)

    G_clients = filter(
        lambda x: x[1].Contrato != NOGARANTIZADO, enumerate(params.clients_vector))

    assign_power_plant(G_clients, True)

    if useNG:
        NG_clients = filter(
            lambda x: x[1].Contrato == NOGARANTIZADO, enumerate(params.clients_vector))
        assign_power_plant(NG_clients, False)

    gain = calculate_gains(params, remaining_energies, client_power_plant)
    non_assigned_clients = sum(map(lambda x: x == -1, client_power_plant))
    return StateRepresentation(params, client_power_plant, remaining_energies, real_consumption, gain, non_assigned_clients)


def generate_simple_initial_state2(params: ProblemParameters, worst=False):
    remaining_energies, real_consumption = generate_state_vars(params)
    client_power_plant = list()
    numClients = len(params.clients_vector)
    numPwP = len(params.power_plants_vector)
    id_client = 0
    id_PwP = 0
    cycle = 0
    MAX_NUM_CYCLE = 15 * numClients / numPwP  # Max num
    total_energy_consumed = 0
    while id_client < numClients:
        if params.clients_vector[id_client].Tipo == NOGARANTIZADO:
            client_power_plant.append(NONPOWERPLANT)

        elif remaining_energies[id_PwP] > real_consumption[id_PwP][id_client]:
            total_energy_consumed += real_consumption[id_PwP][id_client]
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

    client_power_plant = np.array(client_power_plant, dtype=int)

    gain = calculate_gains(params, remaining_energies, client_power_plant)

    non_assigned_clients = 0
    energy_not_suplied = 0
    entropy = calculate_entropy(params, remaining_energies)
    total_remaining_energy = sum(remaining_energies)
    for client, PwP in enumerate(client_power_plant):

        if PwP == -1:
            non_assigned_clients += 1
            energy_not_suplied += params.clients_vector[client].Consumo

    return StateRepresentation(params, client_power_plant, remaining_energies, real_consumption, gain, non_assigned_clients, energy_not_suplied, entropy, total_energy_consumed, total_remaining_energy)

# State Representation


class StateRepresentation(object):
    def __init__(self, params: ProblemParameters, client_power_plant, remaining_energies, real_consumption, gain: float, non_assigned_clients: int, energy_not_supplied: float, entropy: float, total_energy_consumed: float, total_remaining_energy: float):
        self.params = params
        self.client_power_plant = client_power_plant
        self.remaining_energies = remaining_energies
        self.real_consumption = real_consumption

        # Heuristic variables
        self.gain = gain
        self.non_assigned_clients = non_assigned_clients
        self.energy_not_supplied = energy_not_supplied
        self.entropy = entropy
        self.total_energy_consumed = total_energy_consumed
        self.total_remaining_energy = total_remaining_energy

    def copy(self):
        return StateRepresentation(self.params, self.client_power_plant.copy(), self.remaining_energies.copy(), self.real_consumption, self.gain, self.non_assigned_clients, self.energy_not_supplied, self.entropy, self.total_energy_consumed, self.total_remaining_energy)

    def __repr__(self) -> str:
        clientes_servidos = 0
        for i in self.client_power_plant:
            if i != -1:
                clientes_servidos += 1
        return f"client_power_plant: {self.client_power_plant}\n\nClientes servidos = {clientes_servidos}"

    def generate_actions(self) -> Generator[Operator, None, None]:
        numClients = len(self.params.clients_vector)
        numCentrals = len(self.params.power_plants_vector)
        useMergePwP = False
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

                    if id_PwP1 == NONPOWERPLANT or id_PwP2 == NONPOWERPLANT:
                        continue

                    csm_pwp1_cli1 = self.real_consumption[id_PwP1][id_client1]
                    csm_pwp2_cli1 = self.real_consumption[id_PwP2][id_client1]

                    csm_pwp1_cli2 = self.real_consumption[id_PwP1][id_client2]
                    csm_pwp2_cli2 = self.real_consumption[id_PwP2][id_client2]

                    remain1 = self.remaining_energies[id_PwP1]
                    remain2 = self.remaining_energies[id_PwP2]

                    if csm_pwp1_cli2 - csm_pwp1_cli1 < remain1 and csm_pwp2_cli1 - csm_pwp2_cli2 < remain2:
                        yield SwapClients(id_client1, id_client2)

            # Merge Power Plants
            if useMergePwP:
                for id_PwP1 in range(numCentrals):
                    for id_PwP2 in range(id_PwP1 + 1, numCentrals):

                        prod1 = self.params.power_plants_vector[id_PwP1].Produccion
                        prod2 = self.params.power_plants_vector[id_PwP2].Produccion

                        if prod1 >= prod2:
                            bigger = id_PwP1
                            smaller = id_PwP2
                        else:
                            bigger = id_PwP2
                            smaller = id_PwP1

                        consumed = 0

                        for client in range(len(self.params.clients_vector)):
                            if self.client_power_plant[client] == smaller:
                                consumed += self.real_consumption[bigger][client]

                        remaining_bigger = self.remaining_energies[bigger]

                        if remaining_bigger > consumed:
                            yield MergePwPs(smaller, bigger, consumed)

    def generate_one_action(self) -> Generator[Operator, None, None]:
        move_client_combinations = set()
        swap_client_combinations = set()

        numClients = len(self.params.clients_vector)
        numCentrals = len(self.params.power_plants_vector)

        for id_client1 in range(numClients):

            # MoveClient
            for id_PwP in range(numCentrals):
                if self.client_power_plant[id_client1] == id_PwP:
                    continue

                csm_client = self.real_consumption[id_PwP][id_client1]

                if csm_client < self.remaining_energies[id_PwP]:
                    move_client_combinations.add((id_client1, id_PwP))

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
                        swap_client_combinations.add((id_client1, id_client2))

        n = len(move_client_combinations)
        m = len(swap_client_combinations)
        random_value = random.random()
        if random_value < (n / (n + m)):
            combination = random.choice(list(move_client_combinations))
            yield MoveClient(combination[0], combination[1])

        else:
            combination = random.choice(list(swap_client_combinations))
            yield SwapClients(combination[0], combination[1])

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

            if id_PwP1 != NONPOWERPLANT:
                consumption = self.real_consumption[id_PwP1][id_client]
                new_state.remaining_energies[id_PwP1] += consumption
                new_state.total_remaining_energy += consumption

                # La central pasa a estar apagada (sumar el coste de tenerla encendida y restar el de que este apagada).
                # Solo puede pasar si el cliente estaba asignado a alguna central.
                if new_state.remaining_energies[id_PwP1] == PwP1.Produccion:
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
            if new_state.remaining_energies[id_PwP2] == PwP2.Produccion:
                self.gain -= VEnergia.costs_production_mw(PwP2.Tipo) * PwP2.Produccion + VEnergia.daily_cost(PwP2.Tipo) \
                    + VEnergia.stop_cost(PwP2.Tipo)

            consumption = self.real_consumption[id_PwP2][id_client]
            new_state.remaining_energies[id_PwP2] -= consumption
            new_state.total_remaining_energy -= consumption

        elif isinstance(action, SwapClients):
            id_client1 = action.id_client1
            id_client2 = action.id_client2

            id_PwP1 = self.client_power_plant[id_client1]
            id_PwP2 = self.client_power_plant[id_client2]

            new_state.client_power_plant[id_client1] = id_PwP2
            new_state.client_power_plant[id_client2] = id_PwP1

            cons_pp1_c1 = self.real_consumption[id_PwP1][id_client1]
            cons_pp1_c2 = self.real_consumption[id_PwP1][id_client2]
            cons_pp2_c2 = self.real_consumption[id_PwP2][id_client2]
            cons_pp2_c1 = self.real_consumption[id_PwP2][id_client1]

            new_state.remaining_energies[id_PwP1] += cons_pp1_c1 - cons_pp1_c2

            new_state.remaining_energies[id_PwP2] += cons_pp2_c2 - cons_pp2_c1

            new_state.total_remaining_energy += cons_pp1_c1 - \
                cons_pp1_c2 + cons_pp2_c2 - cons_pp2_c1

        elif isinstance(action, RemoveNGClient):
            new_state.non_assigned_clients -= 1
            id_client = action.id_client

            new_state.client_power_plant[id_client] = -1

            id_PwP = self.client_power_plant[id_client]

            new_state.remaining_energies[id_PwP] += self.real_consumption[id_PwP][id_client]

        elif isinstance(action, MergePwPs):
            smaller = action.id_smaller
            bigger = action.id_bigger

            # Relocation of all customers to the largest power station
            new_state.remaining_energies[bigger] -= action.consumed

            # Closure of the smallest power plant
            new_state.remaining_energies[smaller] = self.params.power_plants_vector[smaller].Produccion

            for id_client, PwP in self.client_power_plant:
                if PwP == smaller:
                    new_state.client_power_plant[id_client] = bigger

        return new_state

    def apply_actions_without_copy(self, action: Operator):

        Factores que queremos máximizar:
        - Evitar tener clientes sin asignar -> energia que falta por asignar

        - El uso eficiente de la energia: Intercambiar dos clientes cuando se pueda hacer un mejor uso de energia

        - Llenar las centrales al máximo
                Como hacer para que no hayan ciclos?
                Mirar los cambios de entropia?

        - Priorizar el llenar las centrales de más grandes a las más pequeñas
            En el estado inicial, se cogen primero las centrales del tipo más grande, pero eso no garantiza que se
            coja la central más grande primero

        - Máximizar beneficios

        Variables que usaremos por prioridad:
        - Número de centrales grandes abiertas, Número de centrales medianas abiertas(lo vamos ha
        obviar porque en teoria el generador de estado inicial se ocupa de ello)
        - Clientes atendidos(suma)
        - Energia consumida en bruto(se mide la diferencia, si se consume más, esta resta)
        - Entropia entre energia producida y energia consumida

        - Energia que falta por asignar
        - total remain energy

        Situaciones en que máximizar estos factores puede hacer fallar nuestro modelo:

        """

        return calculate_entropy(self.params, self.remaining_energies)

    def bad_heuristic(self) -> float:
        return calculate_gains(self.params, self.remaining_energies, self.client_power_plant)

    def heuristic_simple_entropy(self) -> float:
        numNGClient = sum(
            map(lambda x: x.Tipo == NOGARANTIZADO, self.params.clients_vector))

        if numNGClient > 10:
            pass
        """entropy = 0
        for id_central, power_plant in enumerate(self.params.power_plants_vector):
            remain = self.remaining_energies[id_central]
            max_prod = power_plant.Produccion

            occupancy = 1 - ((max_prod - remain) / max_prod)
            if occupancy > 0:
                entropy -= occupancy * math.log(occupancy)
        return entropy"""
        return -1


# Clase del problema


class EnergyProblem(Problem):
    def __init__(self, initial_state: StateRepresentation):
        super().__init__(initial_state)

    def actions(self, state: StateRepresentation) -> Generator[Operator, None, None]:
        print(state.gain)
        print(state.better_heuristic2())
        print(sum(state.remaining_energies))
        print(state.remaining_energies)
        print(state.non_assigned_clients)
        return state.generate_actions()

    def result(self, state: StateRepresentation, action: Operator) -> StateRepresentation:
        return state.apply_action(action)

    def value(self, state: StateRepresentation) -> float:
        return state.better_heuristic2()

    # def value(self, state: StateRepresentation) -> float:
    #     return state.better_heuristic()

    def goal_test(self, state: StateRepresentation) -> bool:
        return False


# Ejecución del programa.
clientes = Clientes(ncl=100, propc=[0.4, 0.3, 0.3], propg=1, seed=44)
centrales = Centrales(centrales_por_tipo=[20, 10, 10], seed=44)
parametros = ProblemParameters(
    clients_vector=clientes, power_plants_vector=centrales)
estado_inicial = generate_simple_initial_state2(params=parametros, worst=False)

# clientes = Clientes(ncl=1000, propc=[0.25, 0.3, 0.45], propg=0.75, seed=1234)
# centrales = Centrales(centrales_por_tipo=[5, 10, 25], seed=1234)
# parametros = ProblemParameters(clients_vector=clientes, power_plants_vector=centrales)
# estado_inicial = generate_simple_initial_state(params=parametros)

# print(estado_inicial)
print(estado_inicial.better_heuristic())
print(estado_inicial.gain)
print(estado_inicial.non_assigned_clients)
print()

start = time.time()
ejecucion = hill_climbing(EnergyProblem(estado_inicial))
print(ejecucion)
print(ejecucion.better_heuristic())
print(ejecucion.gain)

print(time.time() - start)

m = ejecucion.client_power_plant
b = 0
for x in m:
    if x == -1:
        b += 1
print(b)
