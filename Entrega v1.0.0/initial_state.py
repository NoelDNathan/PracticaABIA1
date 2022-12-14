from state_representation import *
import numpy as np


def distance(obj1, obj2):
    x = obj1.CoordX - obj2.CoordX
    y = obj1.CoordY - obj2.CoordY
    return (x**2 + y ** 2)**(1/2)


def electry_supplied_to_client(client: Cliente, power_plant: Central) -> int:
    dist = distance(client, power_plant)
    return client.Consumo * (1 + VEnergia.loss(dist))


class InitialState(object):
    @staticmethod
    def empty_state(params: ProblemParameters):
        remain, consum = InitialState.remain_consum(params)
        n = len(params.clients_vector)
        c_pp = [NONPOWERPLANT for _ in range(n)]
        gain, prices = InitialState.calculate_gain(
            params, c_pp, remain,  are_clients_assigned=False)

        c_guaranteed = filter(
            lambda client: client[1].Contrato == GARANTIZADO, enumerate(params.clients_vector))

        misplaced_clients = sum(
            map(lambda client: c_pp[client[0]] == NONPOWERPLANT, c_guaranteed))

        return StateRepresentation(params, c_pp, remain, consum, gain, prices, misplaced_clients=misplaced_clients)

    @staticmethod
    def simple_state(params, asc=True, sort_clients=False) -> StateRepresentation:
        remain, consum = InitialState.remain_consum(params)
        c_pp = []
        pp = list(range(len(params.power_plants_vector)))
        if asc:
            pp.reverse()

        id_pp = pp.pop()
        clients = enumerate(params.clients_vector)
        if sort_clients:
            clients = sorted(enumerate(params.clients_vector),
                             key=lambda x: x[1].Consumo, reverse=True)

        for id_client, client in clients:
            cycles = 0
            if client.Contrato == NOGARANTIZADO:
                c_pp.append(NONPOWERPLANT)
                continue

            c_consum = consum[id_pp][id_client]
            while True:
                if c_consum < remain[id_pp]:
                    c_pp.append(id_pp)
                    remain[id_pp] -= c_consum
                    break
                id_pp = pp.pop()
                if len(pp) == 0:
                    cycles += 1
                    if cycles >= 2:
                        print(remain)
                        print(c_consum)
                        assert(False, "Deberia fallar aqui")
                        print("Not enough power plants to generate initial state")
                        exit()
                    pp = list(range(len(params.power_plants_vector)))
                    if asc:
                        pp.reverse()

        c_pp = np.array(c_pp)

        gain, prices = InitialState.calculate_gain(params, c_pp, remain)

        return StateRepresentation(params, c_pp, remain, consum, gain, prices)

    @staticmethod
    def remain_consum(params):
        remain = []
        consum = []

        for pp in params.power_plants_vector:
            remain.append(pp.Produccion)
            row = []
            for client in params.clients_vector:
                c_consum = electry_supplied_to_client(client, pp)
                row.append(c_consum)
            consum.append(row)

        remain = np.array(remain)
        consum = np.array(consum)
        return remain, consum

    @staticmethod
    def calculate_gain(params, c_pp, remain, all_pp_turned_on=False, are_clients_assigned=True) -> float:
        gain = 0
        prices = [[], [], []]

        for id_pp, pp in enumerate(params.power_plants_vector):
            prices[0].append(VEnergia.stop_cost(pp.Tipo))
            prices[1].append(VEnergia.daily_cost(
                pp.Tipo) + VEnergia.costs_production_mw(pp.Tipo) * pp.Produccion)

            if all_pp_turned_on or pp.Produccion == remain[id_pp]:
                gain -= prices[0][id_pp]
            else:
                gain -= prices[1][id_pp]

        for id_c, c in enumerate(params.clients_vector):
            if c.Contrato == GARANTIZADO:
                if are_clients_assigned:
                    gain += c.Consumo * \
                        VEnergia.tarifa_cliente_garantizada(c.Tipo)
            else:
                gain += c.Consumo * \
                    VEnergia.tarifa_cliente_no_garantizada(c.Tipo)

            prices[2].append(
                c.Consumo * VEnergia.tarifa_cliente_penalizacion(c.Tipo))
            if c_pp[id_c] == NONPOWERPLANT:
                gain -= prices[2][id_c]

        return gain, prices
