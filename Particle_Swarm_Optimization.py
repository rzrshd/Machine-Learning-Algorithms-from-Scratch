import numpy as np
import copy


class ParticleSwarmOptimizedNN():
    def __init__(self, population_size, model_builder, inertia_weight=0.8, cognitive_weight=2, social_weight=2, max_velocity=20):
        self.population_size = population_size
        self.model_builder = model_builder
        self.best_individual = None
        self.cognitive_w = cognitive_weight
        self.inertia_w = inertia_weight
        self.social_w = social_weight
        self.min_v = -max_velocity
        self.max_v = max_velocity

    def _build_model(self, id):
        model = self.model_builder(n_inputs=self.X.shape[1], n_outputs=self.y.shape[1])
        model.id = id
        model.fitness = 0
        model.highest_fitness = 0
        model.accuracy = 0
        model.best_layers = copy.copy(model.layers)

        model.velocity = []
        for layer in model.layers:
            velocity = {'W': 0, 'w0': 0}
            if hasattr(layer, 'W'):
                velocity = {'W': np.zeros_like(layer.W), 'w0': np.zeros_like(layer.w0)}

            model.velocity.append(velocity)

        return model

    def _initialize_population(self):
        self.population = []
        for i in range(self.population_size):
            model = self._build_model(id=i)
            self.population.append(model)

    def _update_weights(self, individual):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        for i, layer in enumerate(individual.layers):
            if hasattr(layer, 'W'):
                first_term_W = self.inertia_w * individual.velocity[i]['W']
                second_term_W = self.cognitive_w * r1 * (individual.best_layers[i].W - layer.W)
                third_term_W = self.social_w * r2 * (self.best_individual.layers[i].W - layer.W)
                new_velocity = first_term_W + second_term_W + third_term_W
                individual.velocity[i]['W'] = np.clip(new_velocity, self.min_v, self.max_v)

                first_term_w0 = self.inertia_w * individual.velocity[i]['w0']
                second_term_w0 = self.cognitive_w * r1 * (individual.best_layers[i].w0 - layer.w0)
                third_term_w0 = self.social_w * r2 * (self.best_individual.layers[i].w0 - layer.w0)
                new_velocity = first_term_w0 + second_term_w0 + third_term_w0
                individual.velocity[i]['w0'] = np.clip(new_velocity, self.min_v, self.max_v)

                individual.layers[i].W += individual.velocity[i]['W']
                individual.layers[i].w0 += individual.velocity[i]['w0']

    def _calculate_fitness(self, individual):
        loss, acc = individual.test_on_batch(self.X, self.y)
        individual.fitness = 1 / (loss + 1e-8)
        individual.accuracy = acc

    def evolve(self, X, y, n_generations):
        self.X = X
        self.y = y

        self._initialize_population()

        self.best_individual = copy.copy(self.population[0])

        for epoch in range(n_generations):
            for individual in self.population:
                self._update_weights(individual)
                self._calculate_fitness(individual)

                if individual.fitness > individual.highest_fitness:
                    individual.best_layers = copy.copy(individual.layers)
                    individual.highest_fitness = individual.fitness

                if individual.fitness > self.best_individual.fitness:
                    self.best_individual = copy.copy(individual)

            print(f'{epoch} Best Individual - ID: {self.best_individual.id} Fitness: {self.best_individual.fitness}, Accuracy: {100 * float(self.best_individual.accuracy)}')

        return self.best_individual
