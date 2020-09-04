import torch

# Minor refactor from https://discourse.numenta.org/t/ive-implemented-htm-to-run-on-gpu-with-pytorch-and-made-it-to-generate-text/7458

class SpatialPooler:
    def __init__(self, input_size, columns, active_columns, device='cpu'):
        self.input_size = input_size
        self.columns = columns
        self.active_columns = active_columns

        self.sparsity = self.active_columns / self.columns

        self.boosting_intensity = 10.0
        self.duty_cycle_inertia = 0.99

        self.permanence_threshold = 0.0
        self.permanence_increment = 0.1
        self.permanence_decrement = 0.05

        self.activation = torch.zeros(self.columns, dtype=torch.bool, device=device)
        self.overlaps = torch.zeros(self.columns, dtype=torch.long, device=device)
        self.duty_cycle = torch.zeros(self.columns, dtype=torch.float32, device=device)

        self.active = torch.zeros(self.active_columns, dtype=torch.long, device=device)

        self.permanence = torch.randn(self.columns, self.input_size, device=device)

    def reset(self):
        self.duty_cycle = torch.zeros(self.columns, dtype=torch.float32, device=device)

    def run(self, input, learning=True):
        weight = self.permanence > self.permanence_threshold
        self.overlaps = torch.sum(input & weight, axis=1)
        
        boosting = torch.exp(self.boosting_intensity * (self.sparsity - self.duty_cycle))
        sorted = (boosting * self.overlaps).argsort()
        self.active = sorted[-self.active_columns:]

        self.activation[:] = False
        self.activation[self.active] = True

        self.duty_cycle *= self.duty_cycle_inertia
        self.duty_cycle[self.active] += 1.0 - self.duty_cycle_inertia

        if learning:
            self.permanence[self.active] += input * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement
        
class TemporalMemory:
    def __init__(self, columns, cells, device='cpu'):
        self.device = device

        self.columns = columns
        self.cells = cells

        self.segment_active_threshold = 10
        self.segment_matching_threshold = 10

        self.synapse_sample_size = 20

        self.permanence_initial = 0.2
        self.permanence_threshold = 0.5
        self.permanence_increment = 0.1
        self.permanence_decrement = 0.05
        self.permanence_punishment = 0.01

        self.cell_active = torch.zeros((self.columns, self.cells), dtype=torch.bool, device=device)
        self.cell_predictive = torch.zeros_like(self.cell_active)
        self.cell_segments = torch.zeros((self.columns, self.cells), dtype=torch.long, device=device)

        self.segment_capacity = 1
        self.segment_index = torch.arange(self.cells * self.segment_capacity, dtype=torch.long, device=device).reshape(1, self.cells, self.segment_capacity)
        self.segment_activation = torch.zeros((self.columns, self.cells, self.segment_capacity), dtype=torch.long, device=device)
        self.segment_potential = torch.zeros_like(self.segment_activation)
        self.segment_active = torch.zeros((self.columns, self.cells, self.segment_capacity), dtype=torch.bool, device=device)
        self.segment_matching = torch.zeros_like(self.segment_active)
        self.segment_synapses = torch.zeros((self.columns, self.cells, self.segment_capacity), dtype=torch.long, device=device)

        self.cell_synapse_capacity = 0
        self.cell_synapse_cell = torch.full((self.columns, self.cells, self.cell_synapse_capacity), -1, dtype=torch.long, device=device)

        self.segment_synapse_capacity = 1
        self.segment_synapse_cell = torch.full((self.columns, self.cells, self.segment_capacity, self.segment_synapse_capacity), -1, dtype=torch.long, device=device)
        self.segment_synapse_permanence = torch.full((self.columns, self.cells, self.segment_capacity, self.segment_synapse_capacity), self.permanence_initial, dtype=torch.float32, device=device)

        self.prev_winner_cell = torch.zeros(0, dtype=torch.long, device=device)
        self.prev_target_segment = torch.zeros(0, dtype=torch.long, device=device)

    def reset(self):
        self.cell_active = torch.zeros((self.columns, self.cells), dtype=torch.bool, device=device)
        self.cell_predictive = torch.zeros_like(self.cell_active)

        self.segment_activation = torch.zeros((self.columns, self.cells, self.segment_capacity), dtype=torch.long, device=device)
        self.segment_potential = torch.zeros_like(self.segment_activation)
        self.segment_active = torch.zeros((self.columns, self.cells, self.segment_capacity), dtype=torch.bool, device=device)
        self.segment_matching = torch.zeros_like(self.segment_active)

        self.prev_winner_cell = torch.zeros(0, dtype=torch.long, device=device)
        self.prev_target_segment = torch.zeros(0, dtype=torch.long, device=device)

    def run(self, active_column, learning=True):
        cell_predictive = self.cell_predictive[active_column]
        column_bursting = ~torch.any(cell_predictive, axis=1)

        segment_potential = self.segment_potential[active_column].reshape(len(active_column), -1)
        column_best_matching_segment = torch.argmax(segment_potential, axis=1)
        column_least_used_cell = torch.argmin(self.cell_segments[active_column], axis=1)
        column_grow_segment = segment_potential[(torch.arange(len(active_column), dtype=torch.long, device=self.device), column_best_matching_segment)] == 0

        growing_segment_column = torch.nonzero(column_grow_segment, as_tuple=True)[0]
        growing_segment_cell = column_least_used_cell[growing_segment_column]
        winner_cell = cell_predictive.clone()
        winner_cell[(growing_segment_column, growing_segment_cell)] = True
        winner_cell = torch.nonzero(winner_cell, as_tuple=True)
        winner_cell = active_column[winner_cell[0]] * self.cells + winner_cell[1]

        if learning:
            segment_learning = self.segment_active[active_column] | ((self.segment_index == column_best_matching_segment[:, None, None]) & (column_bursting & ~column_grow_segment)[:, None, None])

            learning_segment = torch.nonzero(segment_learning, as_tuple=True)
            learning_segment = active_column[learning_segment[0]] * (self.cells * self.segment_capacity) + learning_segment[1] * self.segment_capacity + learning_segment[2]
            learning_segment_synapse_cell = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[learning_segment]
            learning_segment_synapse_cell_valid = torch.nonzero(learning_segment_synapse_cell >= 0, as_tuple=True)
            learning_segment_synapse_cell = learning_segment_synapse_cell[learning_segment_synapse_cell_valid]
            self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[(learning_segment[learning_segment_synapse_cell_valid[0]], learning_segment_synapse_cell_valid[1])] += self.cell_active.reshape(-1)[learning_segment_synapse_cell] * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement
            
            punished_segment = torch.nonzero(self.segment_active.reshape(-1)[self.prev_target_segment], as_tuple=True)[0]
            punished_segment_synapse_cell = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[punished_segment]
            punished_segment_synapse_cell_valid = torch.nonzero(punished_segment_synapse_cell >= 0, as_tuple=True)
            punished_segment_synapse_cell = punished_segment_synapse_cell[punished_segment_synapse_cell_valid]
            self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[(punished_segment[punished_segment_synapse_cell_valid[0]], punished_segment_synapse_cell_valid[1])] -= self.cell_active.reshape(-1)[punished_segment_synapse_cell] * self.permanence_punishment

            if len(self.prev_winner_cell) > 0:
                growing_segment_column = active_column[growing_segment_column]
                growing_segment = self.cell_segments[(growing_segment_column, growing_segment_cell)]

                max_cell_segments = torch.max(growing_segment) + 1 if len(growing_segment) > 0 else 0
                if max_cell_segments > self.segment_capacity:
                    segment_capacity = max_cell_segments
                    self.segment_index = torch.arange(self.cells * segment_capacity, dtype=torch.long, device=self.device).reshape(1, self.cells, segment_capacity)
                    self.segment_activation = torch.zeros((self.columns, self.cells, segment_capacity), dtype=torch.long, device=self.device)
                    self.segment_potential = torch.zeros_like(self.segment_activation)

                    segment_synapses = torch.zeros((self.columns, self.cells, segment_capacity), dtype=torch.long, device=self.device)
                    segment_synapse_cell = torch.full((self.columns, self.cells, segment_capacity, self.segment_synapse_capacity), -1, dtype=torch.long, device=self.device)
                    segment_synapse_permanence = torch.full((self.columns, self.cells, segment_capacity, self.segment_synapse_capacity), self.permanence_initial, dtype=torch.float32, device=self.device)
                    segment_synapses[:, :, :self.segment_capacity] = self.segment_synapses
                    segment_synapse_cell[:, :, :self.segment_capacity, :] = self.segment_synapse_cell
                    segment_synapse_permanence[:, :, :self.segment_capacity, :] = self.segment_synapse_permanence

                    self.segment_capacity = segment_capacity
                    self.segment_synapses = segment_synapses
                    self.segment_synapse_cell = segment_synapse_cell
                    self.segment_synapse_permanence = segment_synapse_permanence

                learning_segment = torch.cat([learning_segment, growing_segment_column * (self.cells * self.segment_capacity) + growing_segment_cell * self.segment_capacity + growing_segment])
                segment_candidate = torch.sort(torch.cat([self.prev_winner_cell.repeat(len(learning_segment), 1), self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[learning_segment].repeat(1, 2)], axis=1), axis=1).values
                segment_winner_targeted = segment_candidate[:, :-1] == segment_candidate[:, 1:]
                segment_candidate[:, :-1][segment_winner_targeted] = -1
                segment_candidate[:, 1:][segment_winner_targeted] = -1
                segment_index = torch.arange(segment_candidate.shape[0], device=self.device)[:, None]
                candidate_index = torch.arange(segment_candidate.shape[1], device=self.device)
                shuffled_candidate_index = torch.multinomial(torch.ones(1, device=self.device).expand(segment_candidate.shape), segment_candidate.shape[1]) #torch.repeat(candidate_index, (segment_candidate.shape[0], 1))
                #torch.apply_along_axis(torch.random.shuffle, 1, shuffled_candidate_index)
                segment_candidate[:, candidate_index] = segment_candidate[(segment_index, shuffled_candidate_index)]
                
                segment_new_synapses = torch.max(torch.min(self.synapse_sample_size - self.segment_potential.reshape(-1)[learning_segment], torch.sum(segment_candidate >= 0, axis=1)), torch.zeros(1, dtype=torch.long, device=self.device))
                new_synapse_segment = torch.nonzero(segment_new_synapses, as_tuple=True)[0]
                if len(new_synapse_segment) > 0:
                    learning_segment = learning_segment[new_synapse_segment]
                    segment_candidate = segment_candidate[new_synapse_segment]
                    segment_new_synapses = segment_new_synapses[new_synapse_segment]
                    shuffled_candidate_index = shuffled_candidate_index[new_synapse_segment]

                    segment_synapses = self.segment_synapses.reshape(-1)[learning_segment]
                    max_segment_synapses = torch.max(segment_synapses + segment_new_synapses) if len(learning_segment) > 0 else 0
                    if max_segment_synapses > self.segment_synapse_capacity:
                        segment_synapses = torch.zeros(len(learning_segment), dtype=torch.long, device=self.device)
                        valid_segment_synapse = torch.nonzero(self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[learning_segment] > 0, as_tuple=True)
                        segment_synapse_offset = torch.zeros(len(learning_segment), dtype=torch.long, device=self.device)
                        if len(valid_segment_synapse[0]) > 0:
                            valid_segment_synapse_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), 1 + torch.nonzero(valid_segment_synapse[0][1:] != valid_segment_synapse[0][:-1], as_tuple=True)[0]])
                            valid_segment = valid_segment_synapse[0][valid_segment_synapse_offset]
                            segment_synapses[valid_segment] = torch.cat([valid_segment_synapse_offset[1:] - valid_segment_synapse_offset[:-1], len(valid_segment_synapse[0]) - valid_segment_synapse_offset[-1].reshape(1)])
                            segment_synapse_offset[valid_segment] = valid_segment_synapse_offset
                        valid_segment_synapse_target = (valid_segment_synapse[0], torch.arange(len(valid_segment_synapse[0]), dtype=torch.long, device=self.device) - segment_synapse_offset[valid_segment_synapse[0]])
                        valid_segment_synapse = (learning_segment[valid_segment_synapse[0]], valid_segment_synapse[1])
                        self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse_target] = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse]
                        self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse_target] = self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse]

                        max_segment_synapses = torch.max(segment_synapses + segment_new_synapses) if len(learning_segment) > 0 else 0
                        if max_segment_synapses > self.segment_synapse_capacity:
                            segment_synapse_capacity = max_segment_synapses
                            segment_synapse_cell = torch.full((self.columns, self.cells, self.segment_capacity, segment_synapse_capacity), -1, dtype=torch.long, device=self.device)
                            segment_synapse_permanence = torch.zeros((self.columns, self.cells, self.segment_capacity, segment_synapse_capacity), dtype=torch.float32, device=self.device)
                            segment_synapse_cell[:, :, :, :self.segment_synapse_capacity] = self.segment_synapse_cell
                            segment_synapse_permanence[:, :, :, :self.segment_synapse_capacity] = self.segment_synapse_permanence
                            self.segment_synapse_capacity = segment_synapse_capacity
                            self.segment_synapse_cell = segment_synapse_cell
                            self.segment_synapse_permanence = segment_synapse_permanence

                    segment_target = torch.nonzero(segment_candidate >= 0, as_tuple=True)
                    segment_target_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), 1 + torch.nonzero(segment_target[0][1:] != segment_target[0][:-1], as_tuple=True)[0]])
                    segment_target_end = torch.where(segment_new_synapses > 0, segment_target[1][segment_target_offset + segment_new_synapses - 1], torch.zeros(1, dtype=torch.long, device=self.device))
                    segment_new_synapse = torch.arange(len(segment_target[0]), dtype=torch.long, device=self.device) - segment_target_offset[segment_target[0]]
                    segment_target_valid = torch.nonzero(segment_target[1] <= segment_target_end[segment_target[0]], as_tuple=True)
                    segment_target = (segment_target[0][segment_target_valid], segment_target[1][segment_target_valid])
                    segment_new_synapse = segment_synapses[segment_target[0]] + segment_new_synapse[segment_target_valid]

                    segment_target_segment = learning_segment[segment_target[0]]
                    segment_target_candidate = segment_candidate[segment_target]
                    self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[(segment_target_segment, segment_new_synapse)] = segment_target_candidate

                    self.cell_segments[(growing_segment_column, growing_segment_cell)] += 1
                    self.segment_synapses.reshape(-1)[learning_segment] = segment_synapses + segment_new_synapses #+= segment_new_synapses
                    
                    # TODO: they're not candidates at this point
                    candidate_target = (shuffled_candidate_index[segment_target], segment_target[0])
                    candidate_synapse_cell = torch.full((segment_candidate.shape[1], segment_candidate.shape[0]), -1, dtype=torch.long, device=self.device)
                    candidate_synapse_cell[candidate_target] = segment_target_candidate
                    candidate_valid = torch.nonzero(torch.any(candidate_synapse_cell >= 0, axis=1), as_tuple=True)[0]

                    candidate_synapse_cell_candidate = candidate_synapse_cell[candidate_valid]
                    candidate_synapse_cell_candidate_valid = torch.nonzero(candidate_synapse_cell_candidate >= 0, as_tuple=True)
                    candidate_synapse_cell_candidate[(candidate_synapse_cell_candidate_valid[0], 0)] = candidate_synapse_cell_candidate[candidate_synapse_cell_candidate_valid]
                    candidate_synapse_cell_candidate = candidate_synapse_cell_candidate[:, 0]

                    candidate_synapse_cell[candidate_target] = segment_target_segment // self.segment_capacity
                    candidate_synapse_cell = candidate_synapse_cell[candidate_valid]
                    candidate_synapse_cell = torch.cat([candidate_synapse_cell, self.cell_synapse_cell.reshape(self.columns * self.cells, -1)[candidate_synapse_cell_candidate]], axis=1)
                    candidate_synapse_cell = torch.sort(candidate_synapse_cell, axis=1).values
                    candidate_synapse_cell[:, 1:][candidate_synapse_cell[:, 1:] == candidate_synapse_cell[:, :-1]] = -1
                    candidate_synapse_cell_valid = candidate_synapse_cell >= 0

                    candidate_synapses = torch.sum(candidate_synapse_cell_valid, axis=1)
                    max_cell_synapses = torch.max(candidate_synapses)
                    if max_cell_synapses > self.cell_synapse_capacity:
                        cell_synapse_capacity = max_cell_synapses
                        cell_synapse_cell = torch.full((self.columns, self.cells, cell_synapse_capacity), -1, dtype=torch.long, device=self.device)
                        cell_synapse_cell[:, :, :self.cell_synapse_capacity] = self.cell_synapse_cell
                        self.cell_synapse_capacity = cell_synapse_capacity
                        self.cell_synapse_cell = cell_synapse_cell

                    candidate_synapse_cell_valid = torch.nonzero(candidate_synapse_cell_valid, as_tuple=True)
                    candidate_synapse_cell_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), 1 + torch.nonzero(candidate_synapse_cell_valid[0][1:] != candidate_synapse_cell_valid[0][:-1], as_tuple=True)[0]])
                    candidate_synapse_cell_index = torch.arange(len(candidate_synapse_cell_valid[0]), dtype=torch.long, device=self.device) - candidate_synapse_cell_offset[candidate_synapse_cell_valid[0]]
                    candidate_synapse_cell_candidate = candidate_synapse_cell_candidate[candidate_synapse_cell_valid[0]]
                    self.cell_synapse_cell.reshape(-1, self.cell_synapse_capacity)[(candidate_synapse_cell_candidate, candidate_synapse_cell_index)] = candidate_synapse_cell[candidate_synapse_cell_valid]

        cell_active = cell_predictive | column_bursting[:, None]
        self.cell_active[:, :] = False
        self.cell_active[active_column] = cell_active

        active_cell = torch.nonzero(cell_active, as_tuple=True)
        active_cell = (active_column[active_cell[0]], active_cell[1])

        cell_targeted = torch.zeros(self.columns * self.cells, dtype=torch.bool, device=self.device)
        active_cell_synapse_cell = self.cell_synapse_cell[active_cell]
        active_cell_synapse_cell = active_cell_synapse_cell[active_cell_synapse_cell >= 0]
        cell_targeted[active_cell_synapse_cell] = True
        target_cell = torch.nonzero(cell_targeted, as_tuple=True)[0]
        target_segment = torch.nonzero(torch.arange(self.segment_capacity, dtype=torch.long, device=self.device)[None, :] < self.cell_segments.reshape(-1)[target_cell][:, None], as_tuple=True)
        target_segment = target_cell[target_segment[0]] * self.segment_capacity + target_segment[1]
        
        segment_synapse_cell_active = self.cell_active.reshape(-1)[self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[target_segment]]
        segment_synapse_permanence = self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[target_segment]
        segment_synapse_weight = segment_synapse_permanence > self.permanence_threshold

        self.segment_activation[:, :, :] = 0
        self.segment_potential[:, :, :] = 0
        self.segment_activation.reshape(-1)[target_segment] = torch.sum(segment_synapse_cell_active & segment_synapse_weight, axis=1)
        self.segment_potential.reshape(-1)[target_segment] = torch.sum(segment_synapse_cell_active, axis=1)
        self.segment_active = self.segment_activation >= self.segment_active_threshold
        self.segment_matching = self.segment_potential >= self.segment_matching_threshold
        self.cell_predictive = torch.any(self.segment_active, axis=2)

        self.prev_winner_cell = winner_cell
        self.prev_target_segment = target_segment

class HierarchicalTemporalMemory:
    def __init__(self, input_size, columns, cells, active_columns=None, device='cpu'):
        if active_columns is None:
            active_columns = int(columns * 0.02)

        self.spatial_pooler = SpatialPooler(input_size, columns, active_columns, device=device)
        self.temporal_memory = TemporalMemory(columns, cells, device=device)

    def reset(self):
        self.spatial_pooler.reset()
        self.temporal_memory.reset()

    def run(self, input, learning=True):
        self.spatial_pooler.run(input, learning=learning)
        self.temporal_memory.run(self.spatial_pooler.active, learning=learning)


if __name__ == '__main__':
    x = torch.randn([10, 1000]) > 1.0
    model = HierarchicalTemporalMemory(1000, 3200, 32)
    for i in x:
        model.run(i)
        print(model.spatial_pooler.active)