import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn
from tvm.relax.expr_functor import PyExprMutator, mutator

@mutator
class ReluRewriter(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        # TODO: Switch on conv, dense layer
        if call.op.name == "relax.nn.relu":
            # TODO: Replace with operator that verifies the hash.
            return call
            


        return super().visit_call_(call)

######################################################################
# Then we can write a pass to apply the mutator to the whole module.
@tvm.transform.module_pass(opt_level=0, name="ReluToGelu")
class ReluToGelu:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = ReluRewriter(mod)
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = rewriter.visit_expr(func)
                rewriter.builder_.update_func(g_var, func)
        return rewriter.builder_.get()

