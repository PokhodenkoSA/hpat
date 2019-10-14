# from .pio import PIO
from llvmlite import binding
import hpat
import hpat.hiframes
import hpat.hiframes.hiframes_untyped
import hpat.hiframes.hiframes_typed
from hpat.hiframes.hiframes_untyped import HiFramesPass
from hpat.hiframes.hiframes_typed import HiFramesTypedPass
from hpat.hiframes.dataframe_pass import DataFramePass
import numba
import numba.compiler
from numba.compiler import DefaultPassBuilder
from numba import ir_utils, ir, postproc
from numba.targets.registry import CPUDispatcher
from numba.ir_utils import guard, get_definition
from numba.inline_closurecall import inline_closure_call, InlineClosureCallPass
from numba.typed_passes import (NopythonTypeInference, AnnotateTypes, ParforPass, DeadCodeElimination)
from numba.untyped_passes import (DeadBranchPrune, InlineInlinables, InlineClosureLikes)
from hpat import config
from hpat.distributed import DistributedPass
import hpat.io
if config._has_h5py:
    from hpat.io import pio

from numba.compiler_machinery import FunctionPass, register_pass

# workaround for Numba #3876 issue with large labels in mortgage benchmark
binding.set_option("tmp", "-non-global-value-max-name-size=2048")

# this is for previous version of pipeline manipulation (numba hpat_req <0.38)
# def stage_io_pass(pipeline):
#     """
#     Convert IO calls
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     if config._has_h5py:
#         io_pass = pio.PIO(pipeline.func_ir, pipeline.locals)
#         io_pass.run()
#
#
# def stage_distributed_pass(pipeline):
#     """
#     parallelize for distributed-memory
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     dist_pass = DistributedPass(pipeline.func_ir, pipeline.typingctx,
#                                 pipeline.type_annotation.typemap, pipeline.type_annotation.calltypes)
#     dist_pass.run()
#
#
# def stage_df_pass(pipeline):
#     """
#     Convert DataFrame calls
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     df_pass = HiFrames(pipeline.func_ir, pipeline.typingctx,
#                        pipeline.args, pipeline.locals)
#     df_pass.run()
#
#
# def stage_df_typed_pass(pipeline):
#     """
#     Convert HiFrames after typing
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     df_pass = HiFramesTyped(pipeline.func_ir, pipeline.typingctx,
#                             pipeline.type_annotation.typemap, pipeline.type_annotation.calltypes)
#     df_pass.run()
#
#
# def stage_inline_pass(pipeline):
#     """
#     Inline function calls (to enable distributed pass analysis)
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     inline_calls(pipeline.func_ir)
#
#
# def stage_repeat_inline_closure(pipeline):
#     assert pipeline.func_ir
#     inline_pass = InlineClosureCallPass(
#         pipeline.func_ir, pipeline.flags.auto_parallel)
#     inline_pass.run()
#     post_proc = postproc.PostProcessor(pipeline.func_ir)
#     post_proc.run()
#
#
# def add_hpat_stages(pipeline_manager, pipeline):
#     pp = pipeline_manager.pipeline_stages['nopython']
#     new_pp = []
#     for (func, desc) in pp:
#         if desc == 'nopython frontend':
#             # before type inference: add inline calls pass,
#             # untyped hiframes pass, hdf5 io
#             # also repeat inline closure pass to inline df stencils
#             new_pp.append(
#                 (lambda: stage_inline_pass(pipeline), "inline funcs"))
#             new_pp.append((lambda: stage_df_pass(
#                 pipeline), "convert DataFrames"))
#             new_pp.append((lambda: stage_io_pass(
#                 pipeline), "replace IO calls"))
#             new_pp.append((lambda: stage_repeat_inline_closure(
#                 pipeline), "repeat inline closure"))
#         # need to handle string array exprs before nopython rewrites converts
#         # them to arrayexpr.
#         # since generic_rewrites has the same description, we check func name
#         if desc == 'nopython rewrites' and 'generic_rewrites' not in str(func):
#             new_pp.append((lambda: stage_df_typed_pass(
#                 pipeline), "typed hiframes pass"))
#         if desc == 'nopython mode backend':
#             # distributed pass after parfor pass and before lowering
#             new_pp.append((lambda: stage_distributed_pass(
#                 pipeline), "convert to distributed"))
#         new_pp.append((func, desc))
#     pipeline_manager.pipeline_stages['nopython'] = new_pp


def inline_calls(func_ir, _locals):
    work_list = list(func_ir.blocks.items())
    while work_list:
        label, block = work_list.pop()
        for i, instr in enumerate(block.body):
            if isinstance(instr, ir.Assign):
                lhs = instr.target
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == 'call':
                    func_def = guard(get_definition, func_ir, expr.func)
                    if (isinstance(func_def, (ir.Global, ir.FreeVar))
                            and isinstance(func_def.value, CPUDispatcher)):
                        py_func = func_def.value.py_func
                        inline_out = inline_closure_call(
                            func_ir, py_func.__globals__, block, i, py_func,
                            work_list=work_list)

                        # TODO remove if when inline_closure_call() output fix
                        # is merged in Numba
                        if isinstance(inline_out, tuple):
                            var_dict = inline_out[1]
                            # TODO: update '##distributed' and '##threaded' in _locals
                            _locals.update((var_dict[k].name, v)
                                           for k, v in func_def.value.locals.items()
                                           if k in var_dict)
                        # for block in new_blocks:
                        #     work_list.append(block)
                        # current block is modified, skip the rest
                        # (included in new blocks)
                        break

    # sometimes type inference fails after inlining since blocks are inserted
    # at the end and there are agg constraints (categorical_split case)
    # CFG simplification fixes this case
    func_ir.blocks = ir_utils.simplify_CFG(func_ir.blocks)

@register_pass(mutates_CFG=True, analysis_only=False)
class InlinePass(FunctionPass):
    _name = "hpat_inline_pass"

    def __init__(self):
        pass

    def run_pass(self, state):
        inline_calls(state.func_ir, state.locals)
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class InlineClosuresPass(FunctionPass):
    _name = "hpat_inline_closures_pass"

    def __init__(self):
        pass

    def run_pass(self, state):
        assert state.func_ir

        inline_pass = InlineClosureCallPass(state.func_ir,
                                            state.flags.auto_parallel,
                                            state.parfor_diagnostics.replaced_fns,
                                            True)
        inline_pass.run()
        # Remove all Dels, and re-run postproc
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run()
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class PostprocessorPass(FunctionPass):
    _name = "hpat_postprocessor_pass"

    def __init__(self):
        pass

    def run_pass(self, state):
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run()
        return True

class HPATPipeline(numba.compiler.CompilerBase):
    """HPAT compiler pipeline
    """

    def add_pass_in_position(self, pm, pass_cls, position):
        """
        Add a pass to the PassManager after the pass "location"
        """
        assert pm.passes
        pm._validate_pass(pass_cls)
        pm.passes.insert(position, (pass_cls, str(pass_cls)))

        # if a pass has been added, it's not finalized
        pm._finalized = False

    def pass_position(self, pm, location):
        """
        Add a pass to the PassManager after the pass "location"
        """
        assert pm.passes
        pm._validate_pass(location)
        for idx, (x, _) in enumerate(pm.passes):
            if x == location:
                return idx
        else:
            raise ValueError("Could not find pass %s" % location)

    def define_pipelines(self):
        name = 'hpat'
        # pm.create_pipeline(name)
        # this maintains the objmode fallback behaviour
        # if not self.state.flags.force_pyobject:
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)

        position = self.pass_position(pm, InlineInlinables)

        self.add_pass_in_position(pm, InlinePass, position - 1)
        pm.add_pass_after(HiFramesPass, InlinePass)
        # pm.add_pass_after(InlineClosuresPass, HiFramesPass)
        # pm.add_pass_after(HiFramesPass, InlineInlinables)
        # pm.add_pass_after(HiFramesPass, InlineInlinables)
        # pm.add_pass_after(DataFramePass, AnnotateTypes)
        pm.add_pass_after(DataFramePass, AnnotateTypes)
        pm.add_pass_after(PostprocessorPass, AnnotateTypes)
        pm.add_pass_after(HiFramesTypedPass, DataFramePass)
        # pm.add_pass_after(DataFramePass, InlineInlinables)
        # pm.add_pass_after(HiFramesTypedPass, DataFramePass)
        # print(pm.passes)
        pm.add_pass_after(DistributedPass, ParforPass)
        pm.finalize()

        return [pm]

        # self.add_preprocessing_stage(pm)
        # self.add_with_handling_stage(pm)
        # self.add_pre_typing_stage(pm)
        # pm.add_stage(self.stage_inline_pass, "inline funcs")
        # pm.add_stage(self.stage_df_pass, "convert DataFrames")
        # # pm.add_stage(self.stage_io_pass, "replace IO calls")
        # # repeat inline closure pass to inline df stencils
        # pm.add_stage(self.stage_repeat_inline_closure, "repeat inline closure")
        # self.add_typing_stage(pm)
        # # breakup optimization stage since df_typed needs to run before
        # # rewrites
        # # e.g. need to handle string array exprs before nopython rewrites
        # # converts them to arrayexpr.
        # # self.add_optimization_stage(pm)
        # # hiframes typed pass should be before pre_parfor since variable types
        # # need updating, and A.call to np.call transformation is invalid for
        # # Series (e.g. S.var is not the same as np.var(S))
        # pm.add_stage(self.stage_dataframe_pass, "typed dataframe pass")
        # pm.add_stage(self.stage_df_typed_pass, "typed hiframes pass")
        # pm.add_stage(self.stage_pre_parfor_pass, "Preprocessing for parfors")
        # if not self.state.flags.no_rewrites:
        #     pm.add_stage(self.stage_nopython_rewrites, "nopython rewrites")
        # if self.state.flags.auto_parallel.enabled:
        #     pm.add_stage(self.stage_parfor_pass, "convert to parfors")
        # pm.add_stage(self.stage_distributed_pass, "convert to distributed")
        # pm.add_stage(self.stage_ir_legalization,
        #              "ensure IR is legal prior to lowering")
        # self.add_lowering_stage(pm)
        # self.add_cleanup_stage(pm)

    def stage_inline_pass(self):
        """
        Inline function calls (to enable distributed pass analysis)
        """
        # Ensure we have an IR and type information.
        assert self.state.func_ir
        inline_calls(self.state.func_ir, self.state.locals)

    def stage_df_pass(self):
        """
        Convert DataFrame calls
        """
        # Ensure we have an IR and type information.
        assert self.state.func_ir
        df_pass = HiFrames(self.state.func_ir, self.state.typingctx,
                           self.state.args, self.state.locals, self.state.reload_init)
        df_pass.run()

    def stage_io_pass(self):
        """
        Convert IO calls
        """
        # Ensure we have an IR and type information.
        assert self.state.func_ir
        if config._has_h5py:
            io_pass = pio.PIO(self.state.func_ir, self.state.locals)
            io_pass.run()

    def stage_repeat_inline_closure(self):
        assert self.state.func_ir
        inline_pass = InlineClosureCallPass(
            self.state.func_ir, self.state.flags.auto_parallel, typed=True)
        inline_pass.run()
        post_proc = postproc.PostProcessor(self.state.func_ir)
        post_proc.run()

    def stage_distributed_pass(self):
        """
        parallelize for distributed-memory
        """
        # Ensure we have an IR and type information.
        assert self.state.func_ir
        from hpat.distributed import DistributedPass
        dist_pass = DistributedPass(
            self.state.func_ir, self.state.typingctx, self.state.targetctx,
            self.state.type_annotation.typemap, self.state.type_annotation.calltypes,
            self.state.reload_init)
        dist_pass.run()

    def stage_df_typed_pass(self):
        """
        Convert HiFrames after typing
        """
        # Ensure we have an IR and type information.
        assert self.state.func_ir
        df_pass = HiFramesTyped(self.state.func_ir, self.state.typingctx,
                                self.state.type_annotation.typemap,
                                self.state.type_annotation.calltypes)
        df_pass.run()

    def stage_dataframe_pass(self):
        """
        Convert DataFrames after typing
        """
        # Ensure we have an IR and type information.
        assert self.state.func_ir
        df_pass = DataFramePass(self.state.func_ir, self.state.typingctx,
                                self.state.type_annotation.typemap,
                                self.state.type_annotation.calltypes)
        df_pass.run()


# class HPATPipelineSeq(HPATPipeline):
#     """HPAT pipeline without the distributed pass (used in rolling kernels)
#     """

#     def define_pipelines(self, pm):
#         name = 'hpat_seq'
#         pm.create_pipeline(name)
#         self.add_preprocessing_stage(pm)
#         self.add_with_handling_stage(pm)
#         self.add_pre_typing_stage(pm)
#         pm.add_stage(self.stage_inline_pass, "inline funcs")
#         pm.add_stage(self.stage_df_pass, "convert DataFrames")
#         pm.add_stage(self.stage_repeat_inline_closure, "repeat inline closure")
#         self.add_typing_stage(pm)
#         # TODO: dataframe pass needed?
#         pm.add_stage(self.stage_dataframe_pass, "typed dataframe pass")
#         pm.add_stage(self.stage_df_typed_pass, "typed hiframes pass")
#         pm.add_stage(self.stage_pre_parfor_pass, "Preprocessing for parfors")
#         if not self.state.flags.no_rewrites:
#             pm.add_stage(self.stage_nopython_rewrites, "nopython rewrites")
#         if self.state.flags.auto_parallel.enabled:
#             pm.add_stage(self.stage_parfor_pass, "convert to parfors")
#         # pm.add_stage(self.stage_distributed_pass, "convert to distributed")
#         pm.add_stage(self.stage_lower_parfor_seq, "parfor seq lower")
#         pm.add_stage(self.stage_ir_legalization,
#                      "ensure IR is legal prior to lowering")
#         self.add_lowering_stage(pm)
#         self.add_cleanup_stage(pm)

#    def stage_lower_parfor_seq(self):
#        numba.parfor.lower_parfor_sequential(
#            self.state.typingctx, self.state.func_ir, self.state.typemap, self.state.calltypes)

class HPATPipelineSeq(HPATPipeline):
    """HPAT pipeline without the distributed pass (used in rolling kernels)
    """

    def define_pipelines(self):
        name = 'hpat'

        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)

        position = self.pass_position(pm, InlineInlinables)
        if pm.passes[position + 1][0] == DeadBranchPrune:
            position += 1

        self.add_pass_in_position(pm, HiFramesPass, position + 1)
        pm.add_pass_after(InlinePass, InlineInlinables)
        pm.add_pass_after(DataFramePass, AnnotateTypes)
        pm.add_pass_after(HiFramesTypedPass, DataFramePass)
        pm.finalize()

        return [pm]
