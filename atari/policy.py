from rl.policy import Policy, LinearAnnealedPolicy


class DoubleLinearAnnealedPolicy(Policy):
    def __init__(self, inner_policy, attr, value_max1, value_min1, value_max2, value_min2, value_test, nb_steps1,
                 nb_steps2):
        super(DoubleLinearAnnealedPolicy, self).__init__()
        self.first_annealed_policy = LinearAnnealedPolicy(inner_policy, attr, value_max1, value_min1, value_test,
                                                          nb_steps1)
        self.second_annealed_policy = LinearAnnealedPolicy(inner_policy, attr, value_max2, value_min2, value_test,
                                                           nb_steps2)
        self.change_policy_step = nb_steps1

    def get_current_policy(self):
        if self.agent.step == 0:
            self.first_annealed_policy._set_agent(self.agent)
            self.second_annealed_policy._set_agent(self.agent)
        if self.agent.step < self.change_policy_step:
            return self.first_annealed_policy
        else:
            return self.second_annealed_policy

    def get_current_value(self):
        """Return current annealing value
        # Returns
            Value to use in annealing
        """
        return self.get_current_policy().get_current_value()

    def select_action(self, **kwargs):
        """Choose an action to perform
        # Returns
            Action to take (int)
        """
        return self.get_current_policy().select_action(**kwargs)


class CheckpointAnnealedPolicy(DoubleLinearAnnealedPolicy):
    def __init__(self, inner_policy, attr, value_max1, value_min1, value_max2, value_min2, value_test, nb_steps1,
                 nb_steps2, starting_step=0):
        steps1 = max(0, nb_steps1 - starting_step)
        steps2 = max(1, nb_steps2 - starting_step)
        a = -float(value_max1 - value_min1) / float(nb_steps1)
        b = float(value_max1)
        starting_max_1 = max(value_min1, a * float(starting_step) + b)
        a = -float(value_max2 - value_min2) / float(nb_steps2)
        b = float(value_max2)
        starting_max_2 = max(value_min2, a * float(starting_step) + b)

        super(CheckpointAnnealedPolicy, self).__init__(inner_policy, attr, starting_max_1, value_min1, starting_max_2,
                                                       value_min2, value_test, steps1, steps2)