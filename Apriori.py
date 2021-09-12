import numpy as np
import itertools


class Rule():
    def __init__(self, antecedent, concequent, confidence, support):
        self.antecedent = antecedent
        self.concequent = concequent
        self.confidence = confidence
        self.support = support


class Apriori():
    def __init__(self, min_sup=0.3, min_conf=0.81):
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.freq_itemsets = None
        self.transactions = None

    def _calculate_support(self, itemset):
        count = 0
        for transaction in self.transactions:
            if self._transaction_contains_items(transaction, itemset):
                count += 1

        support = count / len(self.transactions)
        return support

    def _get_frequent_itemsets(self, candidates):
        frequent = []
        for itemset in candidates:
            support = self._calculate_support(itemset)
            if support >= self.min_sup:
                frequent.append(itemset)

        return frequent

    def _has_infrequent_itemsets(self, candidate):
        k = len(candidate)
        subsets = list(itertools.combinations(candidate, k - 1))
        for t in subsets:
            subset = list(t) if len(t) > 1 else t[0]
            if not subset in self.freq_itemsets[-1]:
                return True

        return False

    def _generate_candidates(self, freq_itemset):
        candidates = []
        for itemset1 in freq_itemset:
            for itemset2 in freq_itemset:
                # Valid if every element but the last are the same and the last element in itemset1 is smaller than the last in itemset2
                valid = False
                single_item = isinstance(itemset1, int)
                if single_item and itemset1 < itemset2:
                    valid = True
                elif not single_item and np.array_equal(itemset1[:-1], itemset2[:-1]) and itemset1[-1] < itemset2[-1]:
                    valid = True

                if valid:
                    if single_item:
                        candidate = [itemset1, itemset2]
                    else:
                        candidate = itemset1 + [itemset2[-1]]

                    infrequent = self._has_infrequent_itemsets(candidate)
                    if not infrequent:
                        candidates.append(candidate)

        return candidates

    def _transaction_contains_items(self, transaction, items):
        if isinstance(items, int):
            return items in transaction

        for item in items:
            if not item in transaction:
                return False

        return True

    def find_frequent_itemsets(self, transactions):
        self.transactions = transactions
        unique_items = set(item for transaction in self.transactions for item in transaction)
        self.freq_itemsets = [self._get_frequent_itemsets(unique_items)]

        while(True):
            candidates = self._generate_candidates(self.freq_itemsets[-1])
            frequent_itemsets = self._get_frequent_itemsets(candidates)

            if not frequent_itemsets:
                break

            self.freq_itemsets.append(frequent_itemsets)

        frequent_itemsets = [itemset for sublist in self.freq_itemsets for itemset in sublist]
        return frequent_itemsets

    def _rules_from_itemset(self, initial_itemset, itemset):
        rules = []
        k = len(itemset)
        subsets = list(itertools.combinations(itemset, k - 1))
        support = self._calculate_support(initial_itemset)
        for antecedent in subsets:
            antecedent = list(antecedent)
            antecedent_support = self._calculate_support(antecedent)
            confidence = float('{0:.2f}'.format(support / antecedent_support))
            if confidence >= self.min_conf:
                concequent = [itemset for itemset in initial_itemset if not itemset in antecedent]
                if len(antecedent) == 1:
                    antecedent = antecedent[0]
                if len(concequent) == 1:
                    concequent = concequent[0]

                rule = Rule(antecedent=antecedent, concequent=concequent, confidence=confidence, support=support)
                rules.append(rule)

                if k - 1 > 1:
                    rules += self._rules_from_itemset(initial_itemset, antecedent)

        return rules

    def generate_rules(self, transactions):
        self.transactions = transactions
        frequent_itemsets = self.find_frequent_itemsets(transactions)
        frequent_itemsets = [itemset for itemset in frequent_itemsets if not isinstance(itemset, int)]
        rules = []
        for itemset in frequent_itemsets:
            rules += self._rules_from_itemset(itemset, itemset)

        return rules
