"""Fingerprint recording code independently of ordinary GPU measurements."""

import ast
import hashlib
from pathlib import Path


RECORDING_ABI = 'gspim-recording-2'
RECORDING_FILES = (
    'pim_capture.py', 'pim_record.py', 'pim_record_ops.py',
    'pim_record_models.py', 'pim_requests.py', 'pim_trace.py',
    'pipeline.py', 'dataflow.py', 'renderer.py', 'renderer.cu',
)


class _RecordingExecution(ast.NodeTransformer):
    """Specialize dataflow's explicitly supplied execution argument."""

    @staticmethod
    def _model_guard(node):
        # Only discard the known side-effect-free attribute guard before an
        # execution-is-None test. Unknown calls remain fingerprinted.
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'hasattr' and len(node.args) == 2
                and not node.keywords and isinstance(node.args[0], ast.Attribute)
                and isinstance(node.args[0].value, ast.Name)
                and node.args[0].value.id == 'self' and node.args[0].attr == 'model'
                and isinstance(node.args[1], ast.Constant))

    def visit_Compare(self, node):
        if (isinstance(node.left, ast.Name) and node.left.id == 'execution'
                and len(node.ops) == 1 and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value is None):
            if isinstance(node.ops[0], ast.Is):
                return ast.Constant(False)
            if isinstance(node.ops[0], ast.IsNot):
                return ast.Constant(True)
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        is_and = isinstance(node.op, ast.And)
        values = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                if value.value != is_and:
                    if all(self._model_guard(item) for item in values):
                        return ast.Constant(not is_and)
                    return ast.BoolOp(node.op, [*values, value])
            else:
                values.append(value)
        if not values:
            return ast.Constant(is_and)
        return values[0] if len(values) == 1 else ast.BoolOp(node.op, values)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node


class _RecordingPython(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        # GPURecorder supplies execution and overrides render_window. These
        # methods only measure/validate ordinary GPU paths after the capture.
        excluded = {
            'CUDAPipeline': {'baseline', 'evaluate', '_render_window'},
            'CUDARenderer': {'generate_baseline'},
            'Ex4DGSCheckpoint': {'baseline_model'},
        }.get(node.name, set())
        node.body = [item for item in node.body
                     if not isinstance(item, ast.FunctionDef) or item.name not in excluded]
        if node.name == 'CUDAPipeline':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'dataflow':
                    _RecordingExecution().visit(item)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # Documentation does not change recorded services.
        return self.generic_visit(node)


def recording_python(source):
    tree = _RecordingPython().visit(ast.parse(source))
    return ast.dump(tree, include_attributes=False).encode()


def recording_sources(workload, source=None):
    """Return named dependencies, without reading any checkpoint or trace."""
    source = Path(source) if source is not None else Path(__file__).parent
    paths = [source / name for name in RECORDING_FILES]
    paths.append(source.parent / 'integrations' / f'{workload}.py')
    # Keep native compilation inputs byte-exact: even identical RGB does not
    # prove identical GPU service times.
    paths.extend(path for path in sorted((source / 'pim_kernels').iterdir())
                 if path.suffix in {'.cu', '.cuh'})
    result = {}
    for path in paths:
        content = path.read_bytes()
        if path.suffix == '.py':
            content = recording_python(content)
        result[str(path.relative_to(source.parent))] = hashlib.sha256(content).hexdigest()
    return result
