import numpy as np
import random
import math


def run():
    population_working.run()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def norm(array, axis = 0, nullval = None):
    ''' Agents are [meaning, signal]: axis = 0 for reception
                                        axis = 1 for production
        nullval: True for Nulls to Zero,
                    None for Nulls to uniform distribution'''
    axis_length = float(array.shape[axis])
    multiplier = array.sum(axis=axis)
    # the following sets values to -1 as a placemarker if it needs to
    # be converted to a uniform distribution
    if nullval == None:
        multiplier = np.where(multiplier == 0., -1. , multiplier)
    multiplier = np.where(multiplier != 0., 1/multiplier, 0.)
    # Corrects axis for multiplier
    if axis == 1:
        multiplier = multiplier[:,np.newaxis]
    sumthing = 1 / axis_length
    output = np.where(multiplier != -1., array * multiplier, sumthing)
    return output


def pro_weights(agent, function, production):
    if production == 'assoc':
        if function == 'sto':
            output = norm(agent, axis = 1)
        if function == 'wta':
            output = wta_matrix(agent, axis = 1)
    if production == 'obvert':
        if function == 'sto':
            listener = norm(agent, axis = 0)
            output = norm(listener, axis = 1)
        if function == 'wta':
            listener = wta_matrix(agent, axis = 0)
            output = wta_matrix(listener, axis = 1)
    return output


def rec_weights(agent, function, reception):
    if reception == 'assoc':
        if function == 'sto':
            output = norm(agent, axis = 0)
        if function == 'wta':
            output = wta_matrix(agent, axis = 0)
    if reception == 'obvert':
        if function == 'sto':
            listener = norm(agent, axis = 1)
            output = norm(listener, axis = 0)
        if function == 'wta':
            listener = wta_matrix(agent, axis = 1)
            output = wta_matrix(listener, axis = 0)
    return output


def average_ca(pro_matrix, rec_matrix):
    # Cartesian product of production and reception matrix
    summed = np.sum(np.sum(pro_matrix * rec_matrix))
    return summed/ len(pro_matrix)


def pop_ca(pop_productions, pop_receptions):
    # Both ways ca for every pair in population
    scores = []
    for i in range(len(pop_productions)):
        for j in range(i+1,len(pop_receptions)):
            score1 = average_ca(pop_productions[i], pop_receptions[j])
            score2 = average_ca(pop_productions[j], pop_receptions[i])
            scores.append((score1 + score2)/ 2.)
    return sum(scores)/ len(scores)


def pop_ca_new(pop_productions, pop_receptions):
    ca_matrix = pop_ca_matrix(pop_productions, pop_receptions)
    ca = ca_matrix.sum() / len(ca_matrix)
    return ca


def pop_ca_matrix(pop_productions, pop_receptions):
    crossovers = math.pow(len(pop_productions), 2)
    total_cartesian = (pop_productions.sum(axis = 0) * pop_receptions.sum(axis = 0))
    subtractor = pop_productions[0] * pop_receptions[0]
    for i in range(1, len(pop_productions)):
        subtractor += pop_productions[i] * pop_receptions[i]
    total_cartesian -= subtractor
    ca_matrix = total_cartesian / (crossovers - len(pop_productions))
    return ca_matrix


def pop_average(pop_array):
    average = pop_array.sum(axis=0)
    return average / len(pop_array)


def wta_matrix(array, axis = 0):
    # Matrix goes to 1. vals for max, 0. elsewhere
    max_vector = np.amax(array, axis = axis)
    if axis == 1:
        max_vector = max_vector[:,np.newaxis]
    maxes =  np.where(array == max_vector, 1., 0.)
    maxes = norm(maxes, axis)
    return maxes


def roulette(inlist):
    total = np.sum(inlist)
    choice = random.uniform(0., total)
    count = 0.
    for i, val in enumerate(inlist):
        count += val
        if count >= choice:
            return i


def obvert(array, target, function = 'sto', direction = 'pro'):
    if direction == 'pro':
        weights = array[:,target]
    elif direction == 'rec':
        weights = array[target]
    if function == 'sto':
        return roulette(weights)
    elif function == 'wta':
        return random.choice(np.where(weights == weights.max()))

def pop_sample(production_matrices, samples, listed = False):
    pro_mat = np.array(production_matrices)
    output = []
    # Sums production matrices in place to get average
    average_pro = pro_mat.sum(axis = 0)
    average_pro /= pro_mat.shape[0]
    # Multinomial for uniform dist of m-productions
    num_meanings = len(average_pro)
    m_distribution = [1/ float(num_meanings) for _ in range(num_meanings)]
    m_samples = np.random.multinomial(samples, m_distribution)
    # Multinomials for production of each s_distro given pro matrix
    for m, count in enumerate(m_samples):
        signals_for_m = np.random.multinomial(count, average_pro[m])
        output.append(list(signals_for_m))
    if listed == False:
        return np.array(output)
    if listed == True:
        output_list = []
        for i, meaning in enumerate(output):
            for j, val in enumerate(meaning):
                for k in range(val):
                    output_list.append([i,j])
        return output_list


def memory_limit(agent, memlimit):
    # Doesn't behave well when more than one exemplar is added!
    if memlimit == None:
        return agent
    shape = agent.shape
    total = np.sum(agent)
    extra = int(total - memlimit)
    if extra <= 0:
        return agent
    if extra == 1:
        flatagent = single_deletion(agent)
    if extra > 1:
        flatagent = agent.flatten()
        for i in range(extra):
            flatagent = single_deletion(flatagent)
    return flatagent.reshape(shape)


def single_deletion(agent):
    weights = agent.flatten()
    probs = norm(weights)
    deletions = np.random.multinomial(1, probs)
    flatagent = weights - deletions
    return flatagent
































