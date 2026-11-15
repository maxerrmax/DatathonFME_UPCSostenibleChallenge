"""
Agent de diagnòstic (Mode Difícil): no fa res (0 kW totes les hores).
Serveix per saber si el servidor paga l'excedent exportat:
  - si 'Estalvi Cost' surt > 0  -> el servidor remunera l'exportació (p. ex. 0,04 €/kWh)
  - si surt 0 %                 -> exportació a 0 €
"""


class BatteryAgent:
    def __init__(self):
        pass

    def act(self, obs):
        return 0.0
