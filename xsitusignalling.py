# packages
import numpy as np
import numpy
import matplotlib.pyplot as plt
import random
import copy


# my files
import matrix


def run():
    pop = population()
    pop.run_graph()
    return pop


def find_non_optimal():
    for i in range(1000):
        pop = population()
        pop.run()
        print pop.ca_data[-1]
        if pop.ca_data[-1] != 1.0:
            pop.do_graph()
            break


def find_optimal():
    for i in range(1000):
        pop = population()
        pop.run()
        print pop.ca_data[-1], '(', i, ' runs)'
        if pop.ca_data[-1] == 1.0:
            print 'After ', i, ' runs'
            pop.do_graph()
            break


def runop():
    pop = population()
    times = []
    for i in range(100):
        time, x = pop.run_til_optimal()
        times.append(time)
        if i % 10 == 0.: print i, ' runs'
    return times


class population():

    def __init__(self,
                    iterations = 100000,
                    stop_at_optimal = False,
                    num_agents = 10,
                    num_meanings = 5,
                    num_signals = 5,
                    # N E W for XSL #
                    context_size = 3,
                    xsl_obverter_style = 'include', # 'include'/'exclude' non-context
                    xsl_hom_inhib = 'both', # 'in_con', 'out_con', 'both'
                    xsl_add_exemplar = 'many', # 'many', 'one'
                    xsl_min_inhibit = 1, # context_size or 1
                    # XSL complete
                    learning_exposures = 35, # eg. 35
                    memory = 150, # eg. 35 or None
                    dynamic = 'closed', # 'closed' or 'vert'
                    wta_sto = 'wta', # 'wta' or 'sto'
                    inhibition = 'max', # 'min' or 'max'
                    lateral = '', #'h'- homonyms, 's'- synonyms, or 'hs'- both
                    production = 'obvert', # 'assoc' or 'obvert'
                    reception = 'assoc', # 'assoc' or 'obvert'
                    learning = 'observe', # 'observe', 'feed'
                    modify = 'h', # 's'-speaker, 'h'-hearer,'sh'- both
                    punish = '', # 's'-speaker, 'h'-hearer,'sh'- both
                    corrective = False, # True, False
                    reinforce = False, # True, False (impacts start count)
                    lat_inhib_w = 1
                ):
        print lat_inhib_w, 'firstin'
        # Model Parameters
        self.num_agents = num_agents
        self.num_signals = num_signals
        self.num_meanings = num_meanings
        self.context_size = context_size
        # Externals
        self.iterations = iterations
        self.stop_at_optimal = stop_at_optimal
        self.record_first_optimal = False
        self.final_only = False
        # Methods
        self.pop_dynamics= {'closed': self.closed_dynamic,
                            'vert': self.vert_dynamic}
        # Weight Adjustments
        print lat_inhib_w, 'in code lat inhib'
        self.latweight = lat_inhib_w
        self.weights = {'MS': 1,
                        'punish': 1,
                        'lateral': self.latweight,
                        'correction': 1,
                        'speaker_meaning': 1}
        # Model Type
        self.dynamic_name = dynamic
        self.dynamic = self.pop_dynamics[dynamic]
        self.wta_sto = wta_sto
        self.inhibition = inhibition
        self.production = production
        self.reception = reception
        self.memory = memory
        self.modify = modify
        self.punish = punish
        self.corrective = corrective
        self.lateral = lateral
        self.learning = learning
        self.learning_exposures = learning_exposures
        self.reinforce = reinforce
            # New for XSL
        self.xsl_obverter_style = xsl_obverter_style
        self.xsl_hom_inhib = xsl_hom_inhib
        self.xsl_add_exemplar = xsl_add_exemplar
        self.xsl_min_inhibit = xsl_min_inhibit

    def init_data(self):
        # Population Data
        self.population = []
        self.production_matrices = []
        self.reception_matrices = []
        # Result Storage
        self.ca_data = []
        self.first_optimal_iteration = self.iterations
        self.dynamic = self.pop_dynamics[self.dynamic_name]
        if self.dynamic_name == 'vert':
            self.iterations = 1000
        if self.dynamic_name == 'closed':
            self.iterations = self.iterations

    def new_agent(self, ran = False, seed = 10):
        agent = []
        init = 0
        if self.reinforce == True: init = 1
        for i in range(self.num_meanings):
            new_meaning = [init for _ in range(self.num_signals)]
            agent.append(new_meaning)
        agent = numpy.array(agent)
        return agent

    def new_population(self, ran = False):
        self.init_data()
        for i in range(self.num_agents):
            self.population.append(self.new_agent(ran = ran))
        self.population = numpy.array(self.population)
        self.init_matrices()

    def init_matrices(self):
        for agent in self.population:
            self.pro_rec_matrices(agent)
        self.production_matrices = numpy.array(self.production_matrices)
        self.reception_matrices = numpy.array(self.reception_matrices)

    def pro_rec_matrices(self, agent):
        pro_matrix = matrix.pro_weights(agent, self.wta_sto, self.production)
        rec_matrix = matrix.rec_weights(agent, self.wta_sto, self.reception)
        self.production_matrices.append(pro_matrix)
        self.reception_matrices.append(rec_matrix)

    def ca_function(self):
        result = matrix.pop_ca_new(self.production_matrices,
                                 self.reception_matrices)
        self.ca_data.append(result)

    def test_pop_sample(self):
        self.new_population()
        return matrix.pop_sample(self.production_matrices, 50)

    def speak(self, sp_index, meaning, context):
        if self.production == 'obvert' and self.context_size >= 1:
            if self.xsl_obverter_style == 'exclude':
                return self.xsl_obverter_speak(sp_index, meaning, context)
        sp = self.production_matrices[sp_index]
        signal = matrix.roulette(sp[meaning])
        return signal

    def interpret(self, h_index, signal, context):
        if len(context) > 1:
            hr_weights = copy.deepcopy(self.population[h_index])
            out_con = range(self.num_meanings)
            [out_con.remove(i) for i in context]
            for k in out_con:
                hr_weights[k,:] = 0.
            learner_type = self.reception
            hr = matrix.rec_weights(hr_weights, self.wta_sto, learner_type)
        else:
            hr = self.reception_matrices[h_index]
        meaning = matrix.roulette(hr[:, signal])
        return meaning

    def record(self, ag_index, meaning, signal, context, function):
        Flag = False
        agent = self.population[ag_index]
        if function != 'speaker_meaning':
            add_type = self.xsl_add_exemplar
        elif function == 'speaker_meaning':
            add_type = 'one'
        if add_type == 'many':
            for m in context:
                agent[m, signal] += self.weights[function]
        elif add_type == 'one':
            interpretation = self.interpret(ag_index, signal, context)
            agent[interpretation, signal] += self.weights[function]
        agent = self.inhibit(agent, meaning, signal, context)

        # DIAGNOSTIC
        ole_agent = copy.copy(agent)
        k = np.where(agent<0)
        if len(k[0]) != 0: Flag == True
        agent = matrix.memory_limit(agent, self.memory)
        k = np.where(agent<0)
        if len(k[0]) != 0 and Flag == False:
            print ole_agent, 'old'
            print agent, 'new'
        # END DIAGNOSTIC

        self.population[ag_index] = agent
        pro_matrix = matrix.pro_weights(agent, self.wta_sto, self.production)
        rec_matrix = matrix.rec_weights(agent, self.wta_sto, self.reception)
        self.production_matrices[ag_index] = pro_matrix
        self.reception_matrices[ag_index] = rec_matrix

    def xsl_obverter_speak(self, sp_i, meaning, context):
        sp = self.population[sp_i]
        obv_sp = copy.deepcopy(sp)
        for k in context:
            obv_sp[k, :] = 0.
        obv_pro = matrix.pro_weights(obv_sp, self.wta_sto, self.production)
        signal = matrix.roulette(obv_pro[meaning])
        return signal

    def inhibit(self, agent, meaning, signal, context):
        lat_inhib = self.latweight
        if 'h' in self.lateral:

            if self.inhibition == 'max':
                if self.xsl_add_exemplar == 'one':
                    context.remove(meaning)
                    agent[meaning, signal] += lat_inhib
                    if self.xsl_hom_inhib == 'out_con':
                        out_con = copy.copy(context)
                        [out_con.remove(i) for i in context]
                        for k in out_con:
                            agent[k,signal] -= lat_inhib
                    elif self.xsl_hom_inhib == 'in_con':
                        for k in context:
                            agent[k, signal] -= lat_inhib
                    elif self.xsl_hom_inhib == 'both':
                        agent[:,signal] -= lat_inhib
                elif self.xsl_add_exemplar == 'many':
                    for k in context:
                        agent[k, signal] += lat_inhib
                    agent[:,signal] -= lat_inhib
                agent = numpy.where(agent < 0., 0., agent)

            if self.inhibition == 'min':
                choices = copy.deepcopy(agent[:,signal])
                if self.xsl_add_exemplar == 'one':
                    context.remove(meaning)
                    choices[meaning] = 0.
                    if self.xsl_hom_inhib == 'both':
                        pass
                    if self.xsl_hom_inhib == 'out_con':
                        for k in context:
                            choices[k] = 0.
                    if self.xsl_hom_inhib == 'in_con':
                        out_con = copy.copy(context)
                        [out_con.remove(i) for i in context]
                        for k in out_con:
                            choices[k] = 0.
                    if sum(choices) != 0.:
                        choice = matrix.roulette(choices)
                        agent[choice, signal] -= lat_inhib
                #
                elif self.xsl_add_exemplar == 'many':
                    out_con = copy.copy(context)
                    [out_con.remove(i) for i in context]
                    for k in range(self.xsl_min_inhibit):
                        target = random.choice(out_con)

        if 's' in self.lateral:
            if self.inhibition == 'max':
                agent[meaning, signal] += lat_inhib
                agent[meaning] -= lat_inhib
                agent = numpy.where(agent < 0., 0., agent)
            if self.inhibition == 'min':
                choices = copy.deepcopy(agent[meaning])
                choices[signal] = 0.
                if sum(choices) != 0.:
                    choice = matrix.roulette(choices)
                    agent[meaning, choice] -= lat_inhib
        return agent

    def closed_dynamic(self):
        sp_i, hr_i = self.two_indices()
        meaning, context = self.get_meaning_with_context()
        signal = self.speak(sp_i, meaning, context)
        if self.learning == 'observe':
            self.update_agents(sp_i, hr_i, meaning, signal, context, 'MS')
        elif self.learning == 'feed':
            self.feedback(sp_i, hr_i, meaning, signal, context)

    def update_agents(self, speaker_i, hearer_i, meaning, signal, context,
                            function, interp = None):
        if 's' in self.modify:
            if function != 'correction':
                self.record(speaker_i, meaning, signal, context, 'speaker_meaning')
            elif function == 'correction':
                self.record(speaker_i, interp, signal, context, function)
        if 'h' in self.modify:
            self.record(hearer_i, meaning, signal, context, function)

    def two_indices(self):
        indices = range(self.num_agents)
        i = indices.pop(random.randrange(len(indices)))
        j = random.choice(indices)
        return i, j

    def vert_dynamic(self):
        # Hopefully done? NOPE! Maybe?
        self.add_agent_to_pop()
        hr_i = self.num_agents
        for i in range(self.learning_exposures):
            sp_i = random.randrange(self.num_agents)
            meaning, context = self.get_meaning_with_context()
            signal = self.speak(sp_i, meaning, context)
            if self.learning == 'observe':
                self.update_agents(sp_i, hr_i, meaning, signal, context, 'MS')
            if self.learning == 'feed':
                self.feedback(sp_i, hr_i, meaning, signal, context)
        self.delete_oldest_agent()

    def get_meaning_with_context(self):
        cx = range(self.num_meanings)
        context = [cx.pop(random.randrange(len(cx))) \
                        for _ in range(self.context_size)]
        meaning = random.choice(context)
        return meaning, context

    def init_new_agent(self):
        new_agent = self.new_agent()
        pro_mat = matrix.pro_weights(new_agent, self.wta_sto, self.production)
        rec_mat = matrix.rec_weights(new_agent, self.wta_sto, self.reception)
        new_agent = np.array([new_agent])
        pro_mat, rec_mat = np.array([pro_mat]), np.array([rec_mat])
        return new_agent, pro_mat, rec_mat

    def add_agent_to_pop(self):
        new_agent, pro_mat, rec_mat = self.init_new_agent()
        self.population = np.append(self.population, new_agent, 0)
        self.production_matrices = np.append(self.production_matrices, pro_mat, 0)
        self.reception_matrices = np.append(self.reception_matrices, rec_mat, 0)

    def delete_oldest_agent(self):
        self.population = self.population[1:]
        self.production_matrices = self.production_matrices[1:]
        self.reception_matrices = self.reception_matrices[1:]

    def feedback(self, s_index, h_index, meaning, signal, context):
        interpreted_meaning = self.interpret(h_index, signal, context)
        # Hearer points at interpretation, Speaker agrees
        if interpreted_meaning == meaning:
            self.update_agents(s_index, h_index, meaning, signal, context, 'MS')
            return
        # Speaker disagrees. First punish hearer or speaker or both
        self.punish_agents(s_index, h_index, meaning, interpreted_meaning, signal, context)
        # Corrective feedback
        if self.corrective == True:
            self.update_agents(s_index, h_index, meaning, signal, context,
                                'correction', interp = interpreted_meaning)

    def punish_agents(self, s_index, h_index, meaning, interp, signal, context):
        indices = []
        mngs = []
        punishment = self.weights['punish']
        if 's' in self.punish:
            indices.append(s_index)
            # THIS BIT: Reinforcement learners have no access to
            # the interpretation, but feedback learners *do*
            if self.reinforce == False:
                mngs.append(interp)
            else:
                mngs.append(meaning)
        if 'h' in self.punish:
            indices.append(h_index)
            mngs.append(interp)
        for i, val in enumerate(indices):
            ag = self.population[val]
            mng = mngs[i]
            if ag[mng][signal] > 0:
                ag[mng][signal] -= punishment
            self.population[i] = ag
            pro_matrix = matrix.pro_weights(ag, self.wta_sto, self.production)
            rec_matrix = matrix.rec_weights(ag, self.wta_sto, self.reception)
            self.production_matrices[i] = pro_matrix
            self.reception_matrices[i] = rec_matrix

    def run(self):
        self.new_population()
        self.ca_function()
        for i in range(self.iterations):
            self.dynamic()
            if self.final_only == False:
                self.ca_function()
            if self.stop_at_optimal == True:
                if self.ca_data[-1] == 1.0:
                    return i
            if self.record_first_optimal == True:
                if self.ca_data[-1] == 1.0:
                    self.first_optimal_iteration = i
            if type(self.stop_at_optimal) == float:
                if self.ca_data[-1] >= self.stop_at_optimal:
                    return i
            if i % 10000 == 0:
                if i != 0:
                    print i, ' iters'
        if self.final_only == True:
            self.ca_function()
        print self.population[0], self.population[1]


    def run_graph(self, multi=False):
        self.run()
        self.do_graph()

    def do_graph(self):
        print self.population[0], 'pop ag0'
        print self.production_matrices[0], 'pro mat ag0'
        print self.reception_matrices[0], 'rec mat ag0'
        plt.plot(self.ca_data)
        plt.ylim(-.1,1.1)
        plt.show()

    def run_ca_data(self):
        self.run()
        return self.ca_data

    def run_til_optimal(self, maxim=None):
        if maxim == None:
            self.stop_at_optimal = True
        else:
            self.stop_at_optimal = maxim
        stop_time = self.run()
        if stop_time == None:
            stop_time = self.iterations
        return stop_time, self.ca_data

    def run_til_done(self, final_only = False):
        self.record_first_optimal = True
        self.final_only = final_only
        self.run()
        return self.ca_data, self.first_optimal_iteration














