"""Generation publications and selected source IDs for 4D representations."""

import torch

class Ex4DGSCapture:
    """Retain immutable interpolation fields and each compacted window's IDs."""

    def __init__(self, renderer, statistics):
        self.renderer, self.statistics = renderer, statistics
        self.original_generate = renderer.generate
        self.models, self.records, self.loads = [], [], []
        renderer.generate = self.window

    def reset(self):
        if self.statistics.trace_binary is not None:
            self.models.clear()
            self.records.clear()
            self.loads.clear()

    def window(self, model, times):
        result = self.original_generate(model, times)
        if self.statistics.trace_binary is not None:
            self.models.append(model)
            self.records.append(self.statistics.last_generation_publication)
            self.loads.append(self.statistics.last_generation_loads)
        return result

    def source(self):
        model = self.renderer.generation_model
        attributes = {name: getattr(model, name).cpu().contiguous() for name in
            ('xyz', 'displacement', 'rotation', 'scales', 'opacity', 'sh', 'opacity_center')}
        motion, rotation, variance = self.renderer.generation_parameters
        attributes.update(motion=motion.permute(2, 0, 1).cpu().contiguous(),
                          motion_rotation=rotation.T.cpu().contiguous(), opacity_variance=variance.cpu())
        inputs = [dict(source_ids=selected.source_ids.cpu(), count=len(selected.sh),
                       generation_count=len(selected.sh)) for selected in self.models]
        return dict(workload='ex4dgs', attributes=attributes, window_inputs=inputs,
                    temporal_intervals=model.temporal_intervals.cpu(),
                    count=len(model.sh), static_count=len(model.xyz),
                    interval=model.interval, time_shift=model.time_shift)

class ExplicitCapture:
    """Bind explicit-4D generation blocks to each window's selected source IDs."""

    def __init__(self, renderer, statistics):
        self.renderer, self.statistics = renderer, statistics
        self.models, self.records, self.loads = [], [], []
        self.original = renderer.ops.generate_4d
        self.original_generate = renderer.generate
        renderer.ops.generate_4d = self.generate
        renderer.generate = self.window

    def reset(self):
        if self.statistics.trace_binary is not None:
            self.models.clear()
            self.records.clear()
            self.loads.clear()

    def window(self, model, times):
        result = self.original_generate(model, times)
        if self.statistics.trace_binary is not None:
            self.models.append(model)
        return result

    def generate(self, *args):
        statistics = self.statistics
        if not statistics.generation_enabled or statistics.mode is None:
            return self.original(*args)
        binary = statistics.generation_trace_binary if statistics.trace_binary is not None else statistics.generation_binary
        result = binary.generate(*args, statistics.generation_view,
                                 statistics.generation_span, statistics.mode)
        statistics.generated[result[0].data_ptr()] = result[3:]
        statistics.last_generation_publication = result[6]
        statistics.last_generation_loads = result[7]
        if statistics.trace_binary is not None:
            self.records.append(result[6])
            self.loads.append(result[7])
        return result[:3]

    def source(self):
        fields = ('xyz', 'scales4d', 'rotation_l', 'rotation_r', 'temporal_means', 'opacity', 'sh')
        inputs = []
        for model in self.models:
            keyframes = torch.unique(torch.cat((model.frame_times.long() // 20,
                                                model.frame_times.long() // 20 + 1)))
            inputs.append(dict(attributes={key: getattr(model, key).cpu().contiguous() for key in fields},
                source_ids=model.source_ids.cpu(), source_count=model.source_count,
                keyframes=keyframes.cpu(), masks=model.masks.index_select(0, keyframes).cpu(),
                count=len(model.xyz), generation_count=len(model.xyz)))
        return dict(workload='explicit4d', window_inputs=inputs,
                    count=inputs[0]['count'], static_count=0)

class AnchoredCapture:
    """Keep decoder service, binding expansion and post-decode IDs explicit."""

    def __init__(self, renderer, statistics, model, stamp, binary):
        self.renderer, self.statistics, self.model, self.stamp = renderer, statistics, model, stamp
        self.binary = binary
        self.original, self.original_generate = renderer.ops.generate_anchored, renderer.generate
        self.models, self.records, self.loads, self.decoding, self.opacity = [], [], [], [], []
        renderer.ops.generate_anchored, renderer.generate = self.generate, self.window

    def reset(self):
        if self.statistics.trace_binary is not None:
            self.models.clear();self.records.clear();self.loads.clear()
            self.decoding.clear();self.opacity.clear()

    def window(self, model, times):
        if self.statistics.trace_binary is None:
            return self.original_generate(model, times)
        self.models.append((model, times))
        self.decoder_start = self.stamp(times)
        return self.original_generate(model, times)

    def generate(self, *args):
        if self.statistics.trace_binary is None:
            return self.original(*args)
        self.decoding.append((self.decoder_start, self.stamp(args[-2])))
        result = self.binary.run(*args)
        self.records.append(result[3]);self.opacity.append(result[2])
        return result[:3]

    def source(self):
        fields = ('anchor_positions', 'anchor_offsets', 'anchor_features', 'anchor_log_scales', 'anchor_static')
        inputs = []
        for (model, times), opacity in zip(self.models, self.opacity):
            projected = (opacity.cpu() >= 1 / 255).any(0).nonzero().flatten()
            inputs.append(dict(attributes={key: getattr(model,key).cpu().contiguous() for key in fields},
                source_ids=model.source_anchors.cpu(), source_count=self.model.anchor_count,
                intervals=self.model.intervals.cpu(), times=times.cpu(), selection_count=len(model.source_anchors),
                generation_count=len(model.source_anchors)*10, count=len(projected), projected_ids=projected,
                decoder_parameters={name:{key:value.cpu().contiguous() for key,value in getattr(model,name).state_dict().items()}
                    for name in ('opacity_mlp','covariance_mlp','color_mlp')}))
        return dict(workload='anchored', window_inputs=inputs, count=inputs[0]['count'], static_count=0,
                    decoder_phases=[dict(start_ns=int(a.cpu()),end_ns=int(b.cpu())) for a,b in self.decoding],
                    decoded_color_fields=True)
