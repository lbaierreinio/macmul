import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn
from tvm.relax.expr_functor import PyExprMutator, mutator

@mutator
class ReluRewriter(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        # visit the relax.Call expr, and only handle the case when op is relax.nn.relu
        print(call)
        print('---')
        if call.op.name == "relax.nn.relu":
            return relax.op.nn.gelu(call.args[0]) # Compute hash
            


        return super().visit_call_(call)

######################################################################
# Then we can write a pass to apply the mutator to the whole module.
@tvm.transform.module_pass(opt_level=0, name="ReluToGelu")
class ReluToGelu:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = ReluRewriter(mod)
        for g_var, func in mod.functions_items():
            print([g_var, func])
            if isinstance(func, relax.Function):
                func = rewriter.visit_expr(func)
                rewriter.builder_.update_func(g_var, func)
        return rewriter.builder_.get()

