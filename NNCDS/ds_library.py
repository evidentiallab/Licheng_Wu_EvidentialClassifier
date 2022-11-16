##A library that performs calculations in dempster-Schafer theory.

from itertools import chain, combinations
import itertools
import functools
import math


##############################################################################################################


def Check_MassFunc(function):
    # Check whether the function as an argument is a mass function

    @functools.wraps(function)
    def wrapper(*args):

        if len(args) < 2:
            raise TypeError("Insufficient mass function!")

        for i in range(len(args)):
            if not isinstance(args[i], MassFunc):
                raise TypeError("Only mass functions are supported!")
        return function(*args)

    return wrapper


def Check_Compatibility(function):
    ##Check whether the mass function as an argument is compatible with the original mass function

    @functools.wraps(function)
    def wrapper(*args):

        for i in range(len(args)):
            for j in range(i, len(args)):
                if i != j and not args[i].is_compatible(args[j]):
                    raise TypeError("The two mass functions are incompatible!")
        return function(*args)

    return wrapper


def Check_Not_Empty(function):
    ##Checks whether the mass function passed as an argument is null

    @functools.wraps(function)
    def wrapper(*args):

        for massfunc in args:
            if massfunc.is_empty():
                raise TypeError("This function cannot be called by empty mass function")
        return function(*args)

    return wrapper


def Check_Rule(function):
    ##Check whether the rules meet requirements

    @functools.wraps(function)
    def wrapper(*args):
        if not isinstance(args[1], str):
            raise TypeError("Please enter evidence combination method!\n \
                Dempster / Smets / Disjunctive / Yager / Dubois_prade")
        return function(*args)

    return wrapper


#######################################################################################################


class MassFunc(dict):
    class Rule(str):

        ##The classic Dempster's rule of combination.
        Dempster = 'Dempster'

        ##The Smet's rule of combination.
        Smets = 'Smets'

        ##The disjunctive rule of combination.
        Disjunctive = 'Disjunctive'

        ##The Yager's rule of combination.
        Yager = 'Yager'

        ##The Dubois-Prade's rule of combination.
        Dubois_prade = 'Dubois_prade'

        ##The simple average combination rule.
        Average = 'Average'

        ##The Murphy's rule of combination.
        Murphy = 'Murphy'

        ##The rule of combination based on Jousselme distance theory.
        Jousselme = 'Jousselme'

    def __init__(self, source=None):

        ##Initialize the mass function

        if source != None:
            if isinstance(source, dict):
                source = source.items()

            for (h, v) in source:
                self[h] += v

    def __contains__(self, hypo):
        return dict.__contains__(self, MassFunc.frozen(hypo))

    def __getitem__(self, hypo):
        return dict.__getitem__(self, MassFunc.frozen(hypo))

    def __setitem__(self, hypo, value):
        return dict.__setitem__(self, MassFunc.frozen(hypo), value)

    def __delitem__(self, hypo):
        return dict.__delitem__(self, MassFunc.frozen(hypo))

    def __missing__(self, key):
        return 0.0

    @Check_MassFunc
    def __eq__(self, massfunc):

        ##Override ==, determine whether the two mass functions are equal.

        for h, v in self.items():
            if not h in massfunc:
                if v != 0:
                    return False
            if round(massfunc[h], 6) != round(v, 6):
                return False

        for h, v in massfunc.items():
            if not h in self:
                if v != 0:
                    return False
            if round(self[h], 6) != round(v, 6):
                return False

        return True

    @Check_MassFunc
    def __and__(self, massfunc):

        ##Overloading &, a quick operation for mass function fusion.

        return self.combine('Dempster', massfunc)

    @Check_MassFunc
    def __or__(self, massfunc):

        ##Overloading |, a quick operation for mass function fusion.

        return self.combine('Disjunctive', massfunc)

    @Check_MassFunc
    @Check_Compatibility
    def __add__(self, massfunc):

        ##Used to add two mass functions.

        result = self.copy()

        for (h, v) in massfunc.items():
            result[h] += v

        return result

    @Check_MassFunc
    @Check_Compatibility
    def __sub__(self, massfunc):

        ##Used to subtract two mass functions.

        result = self.copy()

        for (h, v) in massfunc.items():
            result[h] -= v

        return result

    def _sum(self):

        ##Calculate the sum of all values.

        s = 0

        for h, v in self.items():
            s += v

        return s

    def is_empty(self):

        ##Check whether it is empty.

        return self._sum() == 0

    def all(self):

        ##Returns an iterator over all subsets of the frame of discernment, including the empty set.

        return power_set(self.frame())

    @Check_MassFunc
    def is_compatible(self, massfunc):

        ##Determine whether the two mass functions are compatible.

        if self.is_empty() or massfunc.is_empty():
            return True
        elif massfunc.core_hypo() <= self.core_hypo():
            return True

        return False

    def union(self):

        ##Returns the union of discernment of the mass function.

        union = set()
        for f in self.discrete_frame():
            union = union | f

        return union

    def frozen(hypo):

        ##Convert the hypothesis to "frozenset" to make it hashable.

        if isinstance(hypo, frozenset):
            return hypo
        else:
            return frozenset(hypo)

    def copy(self):

        ##Creates a copy of the mass function.

        c = MassFunc()

        for k, v in self.items():
            c[k] = v

        return c

    def frame(self):

        ##Returns the frame of discernment of the mass function.

        if not self:
            return frozenset()
        else:
            return frozenset.union(*self.keys())

    def discrete_frame(self):

        ##Discretize the frame of discernment of the mass function.

        discrete_frame = set()

        for e in self.frame():
            discrete_frame.add(frozenset((e,)))

        return discrete_frame

    def pignistic(self):

        ##BPA is converted into a probability measure using the Smets method.

        p = MassFunc()

        for (h, v) in self.items():
            if v > 0.0:
                size = float(len(h))
                for s in h:
                    p[(s,)] += v / size

        return p.normalize()

    @Check_Rule
    def combine(self, rule, *massfunc):

        ##Determine the combine rule and perform the corresponding operation.

        if rule == MassFunc.Rule.Dempster:
            return self.conjunctive(*massfunc)
        elif rule == MassFunc.Rule.Smets:
            return self.smets(*massfunc)
        elif rule == MassFunc.Rule.Disjunctive:
            return self.disjunctive(*massfunc)
        elif rule == MassFunc.Rule.Yager:
            return self.yager(*massfunc)
        elif rule == MassFunc.Rule.Dubois_prade:
            return self.dubois_prade(*massfunc)
        elif rule == MassFunc.Rule.Average:
            return self.average(*massfunc)
        elif rule == MassFunc.Rule.Murphy:
            return self.murphy(*massfunc)
        elif rule == MassFunc.Rule.Jousselme:
            return self.jousselme(*massfunc)
        else:
            raise ValueError("Please enter evidence combination method!\n \
                            Dempster / Smets / Disjunctive / Yager / Dubois_prade / Average / Murphy / Jousselme")

    @Check_MassFunc
    @Check_Not_Empty
    def conjunctive(self, *massfunc):

        ##Use the classic Dempster's rule of combination to combine two or more mass functions.

        def combine_two(m1, m2):

            conjunction = lambda s1, s2: s1 & s2

            combine = MassFunc()

            for (h1, v1) in m1.items():
                for (h2, v2) in m2.items():
                    index = conjunction(h1, h2)
                    combine[index] += v1 * v2

            for (h, v) in combine.items():
                combine[h] = round(combine[h], 6)

            return combine

        combine = combine_two(self, massfunc[0])

        for massfunc in massfunc[1:]:
            combine = combine_two(combine, massfunc)

        return combine.normalize()

    @Check_MassFunc
    @Check_Not_Empty
    def smets(self, *massfunc):

        ##Used the Smets's rule of combination to combine two or more mass functions.

        def combine_two(m1, m2):

            conjunction = lambda s1, s2: s1 & s2

            combine = MassFunc()

            for (h1, v1) in m1.items():
                for (h2, v2) in m2.items():
                    index = conjunction(h1, h2)
                    combine[index] += v1 * v2

            for (h, v) in combine.items():
                combine[h] = round(combine[h], 6)

            return combine

        combine = combine_two(self, massfunc[0])

        for massfunc in massfunc[1:]:
            combine = combine_two(combine, massfunc)

        return combine

    @Check_MassFunc
    @Check_Not_Empty
    def disjunctive(self, *massfunc):

        ##Use the disjunctive rule of combination to combine two or more mass functions.

        def combine_two(m1, m2):

            disjunction = lambda s1, s2: s1 | s2

            combine = MassFunc()

            for (h1, v1) in m1.items():
                for (h2, v2) in m2.items():
                    index = disjunction(h1, h2)
                    combine[index] += v1 * v2

            for (h, v) in combine.items():
                combine[h] = round(combine[h], 6)

            return combine

        combine = combine_two(self, massfunc[0])

        for massfunc in massfunc[1:]:
            combine = combine_two(combine, massfunc)

        return combine.normalize()

    @Check_MassFunc
    @Check_Not_Empty
    def yager(self, *massfunc):

        ##Use the Yager's rule of combination to combine two or more mass functions.

        def combine_two(m1, m2):

            conjunction = lambda s1, s2: s1 & s2

            combine = MassFunc()

            for (h1, v1) in m1.items():
                for (h2, v2) in m2.items():
                    index = conjunction(h1, h2)
                    combine[index] += v1 * v2

            for (h, v) in combine.items():
                combine[h] = round(combine[h], 6)

            return combine

        combine = combine_two(self, massfunc[0])

        for massfunc in massfunc[1:]:
            combine = combine_two(combine, massfunc)

        if combine[frozenset()] >= 0:
            combine[self.union()] = combine[frozenset()]
            del combine[frozenset()]

        return combine

    @Check_MassFunc
    @Check_Not_Empty
    def dubois_prade(self, *massfunc):

        ##Use the Dubois_prade's rule of combination to combine two or more mass functions.

        def combine_two(m1, m2):

            conjunction = lambda s1, s2: s1 & s2
            disjunction = lambda s1, s2: s1 | s2

            combine = MassFunc()

            for (h1, v1) in m1.items():
                for (h2, v2) in m2.items():

                    index_1 = conjunction(h1, h2)
                    combine[index_1] += v1 * v2

                    if not index_1:
                        index_2 = disjunction(h1, h2)
                        combine[index_2] += v1 * v2

            for (h, v) in combine.items():
                combine[h] = round(combine[h], 6)

            del combine[frozenset()]
            return combine

        combine = combine_two(self, massfunc[0])

        for massfunc in massfunc[1:]:
            combine = combine_two(combine, massfunc)

        return combine

    @Check_MassFunc
    @Check_Not_Empty
    def average(self, *massfunc):

        ##Use the simple average combination rule to combine two or more mass functions.

        combine = self.copy()

        for mf in massfunc:
            for h, v in mf.items():
                combine[h] += v

        for h, v in combine.items():
            combine[h] /= len(massfunc) + 1
            combine[h] = round(combine[h], 6)

        return combine

    @Check_MassFunc
    @Check_Not_Empty
    def murphy(self, *massfunc):

        ##Use the Murphy's rule of combination to combine two or more mass functions.

        average = self.average(*massfunc)
        return average.conjunctive(*([average] * len(massfunc)))

    @Check_MassFunc
    @Check_Not_Empty
    def jousselme(self, *massfunc):

        ##Use the rule of combination based on Jousselme distance theory to combine two or more mass functions.

        masses = [self]
        masses.extend(list(massfunc))
        credibility = MassFunc.crd(*masses)
        Jousslme = MassFunc()

        for crd, mass in zip(credibility, masses):
            for h, v in mass.items():
                if h not in Jousslme.focal_hypo():
                    Jousslme[h] = v * crd
                else:
                    Jousslme[h] += v * crd

        return Jousslme.conjunctive(*([Jousslme] * len(massfunc)))

    def bel(self, hypo=None):

        ##Calculate the belief function for each "hypothesis" or the entire belief function.

        if hypo is None:
            return {h: self.bel(h) for h in power_set(self.core_hypo())}

        bel = 0
        hypo = MassFunc.frozen(hypo)

        for (h, v) in self.items():
            if hypo.issuperset(h):
                bel += v

        return round(bel, 6)

    def pl(self, hypo=None):

        ##Calculate the belief plausibility for each "hypothesis" or the entire plausibility function.

        if hypo is None:
            return {h: self.pl(h) for h in power_set(self.core_hypo())}

        pl = 0
        hypo = MassFunc.frozen(hypo)

        for (h, v) in self.items():
            if hypo & h:
                pl += v

        return round(pl, 6)

    def com(self, hypo=None):

        ##Calculate the commonality plausibility for each "hypothesis" or the entire commonality function.

        if hypo is None:
            return {h: self.com(h) for h in power_set(self.core_hypo())}

        com = 0
        hypo = MassFunc.frozen(hypo)

        for (h, v) in self.items():
            if h.issuperset(hypo):
                com += v

        return round(com, 6)

    def difference(self, massfunc):

        ##Calculate the difference between the current mass function and the other one.

        difference = self.copy()

        for h, v in massfunc.items():

            if h in self.focal_hypo():
                difference[h] = self[h] - massfunc[h]
            else:
                difference[h] = -massfunc[h]

        return difference

    @Check_MassFunc
    @Check_Not_Empty
    def dist(self, *massfunc):

        ##Calculate the distance between the current mass function and the other one or more mass functions.

        def distance_two(massfunc):

            difference = self.difference(massfunc)
            matrix = {}

            for d1 in difference:
                matrix[d1] = {}
                for d2 in difference:
                    if (difference[d1] != 0) or (difference[d2] != 0):
                        matrix[d1][d2] = len(d1 & d2) / len(d1 | d2)
                    else:
                        matrix[d1][d2] = 1

            distance = 0
            temp = {}

            for d1 in difference:
                temp[d1] = 0
                for d2 in difference:
                    temp[d1] += difference[d2] * matrix[d1][d2]

            for d1 in difference:
                distance += temp[d1] * difference[d1]
            return math.sqrt(0.5 * distance)

        distance = 0

        for mf in massfunc:
            distance += distance_two(mf)

        return round(distance / len(massfunc), 6)

    @Check_MassFunc
    @Check_Not_Empty
    def sim(self, massfunc):

        ##Calculate the similarity between the current mass function and the other one.

        return round(1 - self.dist(massfunc), 6)

    @Check_MassFunc
    @Check_Not_Empty
    def sup(self, *massfunc):

        ##Calculate the support between the current mass function and the other one or more mass functions.

        sup = 0

        for mf in massfunc:
            sup += self.sim(mf)

        return round(sup, 6)

    @Check_MassFunc
    @Check_Not_Empty
    @Check_Compatibility
    def crd(*massfunc):

        ##Calculate the credibility between the current mass function and the other one or more mass functions.

        supports = []
        for mf in massfunc:
            supports.append(mf.sup(*[x for x in massfunc if x != mf]))
        crd = []
        supportSum = sum(supports)

        for i in range(len(massfunc)):
            crd.append(round(supports[i] / supportSum, 6))

        return crd

    def normalize(self):

        ##Normalizes the mass function.

        if frozenset() in self:
            del self[frozenset()]

        msum = sum(self.values())

        for (h, v) in self.items():
            self[h] = round(v / msum, 6)

        return self

    def map(self, function):

        ##Mapping hypothesis to another hypothesis as specified by the function.

        massfunc = MassFunc()

        for (h, v) in self.items():
            massfunc[self.frozen(function(h))] += v

        return massfunc

    def core_hypo(self):

        ##Return the core of the mass functions.

        focal_hypo = self.focal_hypo()

        if not focal_hypo:
            return frozenset()
        else:
            return frozenset.union(*focal_hypo)

    def focal_hypo(self):

        ##Return a valid hypothesis that the mass value is greater than 0.

        focal_hypo = set()

        for (h, v) in self.items():
            if v > 0:
                focal_hypo.add(h)

        return focal_hypo

    def validate(self):

        ##Validate the hypothesis.

        remove = set()

        for (h, v) in self.items():
            if v == 0.0:
                remove.add(h)

        for h in remove:
            del self[h]

        return self


##################################################################################################################


def power_set(iterable):
    ##Returns the power set of 'set'.

    return map(frozenset, chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1)))

