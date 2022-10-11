from symbol import power
from typing import List, Generator, Set
from abia_energia import *


# Our Methods

def distance(obj1, obj2):
    x = obj1.CoordX - obj2.CoordX
    y = obj1.CoordY - obj2.CoordY
    return (x**2 + y ** 2)**(1/2)


def electry_supplied_to_client(client: Cliente, power_plant: Central) -> int:
    dist = distance(client, power_plant)
    # return client.Consumo * (1 + VEnergia.loss(dist))
    return client.Consumo / (1 - VEnergia.loss(dist))

# Problem params


class ProblemParameters():
    def __init__(self, clients_vector: Clientes, power_plants_vector: Centrales) -> None:
        self.clients_vector = clients_vector
        self.power_plants_vector = power_plants_vector
# Operators


class Operator():
    pass


class MoveClient(Operator):
    def __init__(self, client, destination_PwP):
        self.client = client
        self.destination_PwP = destination_PwP

    def __repr__(self) -> str:
        return f"Client {self.client} has been moved to power plant {self.destination_PwP}"


class SwapClient(Operator):
    def __init__(self, client1, client2):
        self.client1 = client1
        self.client2 = client2

    def __repr__(self) -> str:
        return f"Swap between Client {self.client1} and Client {self.client2}"


class remove_non_guaranteed_client(Operator):
    def __init__(self, p_i: int, p_j: int):
        pass

    def __repr__(self) -> str:
        pass

# Generators initial states


def generate_simple_initial_state(params: ProblemParameters):
    client_power_plant = list()
    remaining_energies = list()

    ids_power_plants = [id for id in range(len(params.power_plants_vector))]

    id_PwP = ids_power_plants.pop()
    PwP = params.power_plants_vector[id_PwP]
    remain = PwP.Produccion

    for client in params.clients_vector:
        if client.Contrato == NOGARANTIZADO:
            continue

        consum = electry_supplied_to_client(client, PwP)
        while True:

            if consum < remain:
                client_power_plant.append(id_PwP)
                remain -= consum
                break

            remaining_energies.append(remain)
            id_PwP = ids_power_plants.pop()
            PwP = params.power_plants_vector[id_PwP]
            remain = PwP.Produccion

    for id in ids_power_plants:
        PwP = params.power_plants_vector[id]
        remaining_energies.append(PwP.Produccion)


def generate_complex_initial_state():
    pass


class StateRepresentation():
    def __init__(self, params: ProblemParameters, client_power_plant: List[int], remaining_energies: List[float]):
        self.params = params
        self.client_power_plant = client_power_plant
        self.remaining_energies = remaining_energies

    def copy(self):
        pass

    def __repr__(self) -> str:
        pass

    def generate_actions(self) -> Generator[Operator, None, None]:
        for id_client1, client1 in enumerate(self.params.clients_vector):
            for id_power_plant, power_plant in enumerate(self.params.power_plants_vector):
                if self.client_power_plant[id_client1] == id_power_plant:
                    continue

                # Podríamos ahorrarnos calcular esto?
                consum_client = electry_supplied_to_client(
                    client1, power_plant)

                if consum_client < self.remaining_energies[id_power_plant]:
                    yield MoveClient(id_client1, id_power_plant)

            for id_client2, client2 in enumerate(self.params.clients_vector):
                if id_client1 == id_client2:
                    continue

                id_PwP1 = self.client_power_plant[client1]
                id_PwP2 = self.client_power_plant[client2]
                if id_PwP1 == id_PwP2:
                    continue

                # Podríamos ahorrarnos calcular esto?
                consum_client1 = electry_supplied_to_client(
                    client1, power_plant)

                consum_client2 = electry_supplied_to_client(
                    client2, power_plant)

                remaining_energy1 = self.remaining_energies[id_PwP1]
                remaining_energy2 = self.remaining_energies[id_PwP2]

                prod1 = self.params.power_plants_vector[id_PwP1]
                prod2 = self.params.power_plants_vector[id_PwP2]

                if (remaining_energy1 - consum_client1 + consum_client2 < prod1
                        and remaining_energy2 - consum_client2 + consum_client1 < prod2):
                    yield SwapClient(id_client1, id_client2)

    def apply_actions(self, action: Operator) -> StateRepresentation:
        new_state = self.copy()

        if isinstance(action, MoveClient):
            client = action.client

            PwP_1 = client.power_plant
            PwP_2 = action.destination_PwP

            client.power_plant.remaining_energy += client.consumption2PowerPlant()
            new_state.destination_PwP.remaining_energy -= client.consumption2PowerPlant()

        elif isinstance(action, SwapClient):
            client1 = action.client1
            client2 = action.client2

        return new_state

    def heuristic():
        pass
