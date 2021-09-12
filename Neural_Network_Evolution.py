import numpy as np
import copy


class Neural_Network_Evolution():
    def __init__(self, population_size, mutation_rate, model_builder):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_builder = model_builder

    def _build_model(self, id):
        model = self.model_builder(n_inputs=self.X.shape[1], n_outputs=self.y.shape[1])
        model.id = id
        model.fitness = 0
        model.accuracy = 0
        return model

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            model = self._build_model(id=np.random.randint(1000))
            self.population.append(model)

    def _mutate(self, individual, var=1):
        for layer in individual.layers:
            if hasattr(layer, 'W'):
                mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=layer.W.shape)
                layer.W += np.random.normal(loc=0, scale=var, size=layer.W.shape) * mutation_mask
                mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=layer.w0.shape)
                layer.w0 += np.random.normal(loc=0, scale=var, size=layer.w0.shape) * mutation_mask

        return individual

    def _inherit_weights(self, child, parent):
        for i in range(len(child.layers)):
            if hasattr(child.layers[i], 'W'):
                child.layers[i].W = parent.layers[i].W.copy()
                child.layers[i].w0 = parent.layers[i].w0.copy()

    def _crossover(self, parent1, parent2):
        child1 = self._build_model(id=parent1.id + 1)
        self._inherit_weights(child1, parent1)
        child2 = self._build_model(id=parent2.id + 1)
        self._inherit_weights(child2, parent2)

        for i in range(len(child1.layers)):
            if hasattr(child1.layers[i], 'W'):
                n_neurons = child1.layers[i].W.shape[1]
                cutoff = np.random.randint(0, n_neurons)
                child1.layers[i].W[:, cutoff:] = parent2.layers[i].W[:, cutoff:].copy()
                child1.layers[i].w0[:, cutoff:] = parent2.layers[i].w0[:, cutoff:].copy()
                child2.layers[i].W[:, cutoff:] = parent1.layers[i].W[:, cutoff:].copy()
                child2.layers[i].w0[:, cutoff:] = parent1.layers[i].w0[:, cutoff:].copy()

        return child1, child2

    def _calculate_fitness(self):
        for individual in self.population:
            loss, acc = individual.test_on_batch(self.X, self.y)
            individual.fitness = 1 / (loss + 1e-8)
            individual.accuracy = acc

    def evolve(self, X, y, n_generations):
        self.X = X
        self.y = y

        self._initialize_population()

        n_winners = int(self.population_size * 0.4)

        n_parents = self.population_size - n_winners

        for epoch in range(n_generations):
            self._calculate_fitness()

            sorted_i = np.argsort([model.fitness for model in self.population])[::-1]
            self.population = [self.population[i] for i in sorted_i]

            fittest_individual = self.population[0]
            print(f'{epoch} Best Individual - Fitness: {fittest_individual.fitness}, Accuracy: {float(100 * fittest_individual.accuracy)}')


            next_population = [self.population[i] for i in range(n_winners)]

            total_fitness = np.sum([model.fitness for model in self.population])

            parent_probabilities = [model.fitness / total_fitness for model in self.population]
            parents = np.random.choice(self.population, size=n_parents, p=parent_probabilities, replace=False)

            for i in np.arange(0, len(parents), 2):
                child1, child2 = self._crossover(parents[i], parents[i+1])
                next_population += [self._mutate(child1), self._mutate(child2)]

            self.population = next_population

        return fittest_individual
