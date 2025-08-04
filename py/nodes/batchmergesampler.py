import operator

import torch
from comfy.samplers import KSAMPLER

from .. import utils
from .base import DumInputTypes, DumLazyInputTypes


class BatchMergeSampler:
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    INPUT_TYPES = DumLazyInputTypes(
        lambda: DumInputTypes()
        .req_sampler()
        .req_field_mode(("horizontal", "vertical"), default="horizontal")
        .req_int_merge_step(default=5, min=1)
        .req_int_hoverlap(default=0, min=0)
        .req_int_voverlap(default=0, min=0)
        .req_selectblend()
        .req_float_blend_strength(default=0.5)
        .opt_sampler_sampler_after()
        .opt_yaml_advanced_plan(),
    )

    @classmethod
    def parse_plan(cls, plan):
        def filter_empty(i):
            temp = (item.strip() if isinstance(item, str) else item for item in i)
            return (item for item in temp if item)

        def parse_item(s):
            ops = []
            while s and not s[0].isdigit():
                c, s = s[0], s[1:]
                if c == "h":
                    ops.append("hflip")
                elif c == "v":
                    ops.append("vflip")
                elif c == "u":
                    ops.append("uroll")
                elif c == "d":
                    ops.append("droll")
                elif c == "l":
                    ops.append("lroll")
                elif c == "r":
                    ops.append("rroll")
                else:
                    raise ValueError("bad op prefix")
            if not s:
                raise ValueError("item empty after stripping op prefixes")
            return {"idx": int(s), "ops": tuple(ops)}

        def parse_cols(s):
            if not s:
                raise ValueError("empty column item")
            return tuple(map(parse_item, s.split(",")))

        def parse_rows(s):
            s = s.strip()
            if not s:
                raise ValueError("empty grid item")
            return tuple(map(parse_cols, s.split(":")))

        def parse_batch(s):
            return tuple(filter_empty(map(parse_rows, filter_empty(s.split(None)))))

        def parse_line(s):
            s = s.strip()
            if not s or s.startswith("#"):
                return None
            if not s.startswith("@"):
                raise ValueError("Plan items must start with @STEP_NUMBER")
            step, *rest = s.split(None, 1)
            if step == "@" or not rest:
                return None
            step = int(step[1:]) - 1
            rest = rest[0].strip()
            if not rest:
                return None
            batch = parse_batch(rest)
            if not batch:
                return None
            return {"step": step, "batch": batch}

        return tuple(
            sorted(
                filter_empty(map(parse_line, plan.split("\n"))),
                key=operator.itemgetter("step"),
            ),
        )

    @classmethod
    def dump_plan(cls, plan):
        print("\nPLAN:")
        for item in plan:
            print(f"  Step {item['step'] + 1}:")
            for bidx, bitem in enumerate(item["batch"]):
                print(f"    Batch {bidx}:")
                for rowidx, row in enumerate(bitem):
                    prettyrow = " | ".join(
                        f"({','.join(col.get('ops', ()))}){col['idx']}"
                        if col.get("ops")
                        else str(col["idx"])
                        for col in row
                    )
                    print(f"      Row {rowidx:>3}: {prettyrow}")

    @classmethod
    def execute_plan_item(
        cls,
        x,
        pi,
        *,
        hoverlap=0,
        voverlap=0,
        blend=torch.lerp,
        blend_strength=0.5,
    ):
        hoverlap = min(x.shape[-1], max(0, hoverlap))
        voverlap = min(x.shape[-2], max(0, voverlap))
        x_chunks = torch.tensor_split(x, x.shape[0])
        bresult = []
        for bitem in pi["batch"]:
            rresult = []
            for ritem in bitem:
                cresult = []
                for citem in ritem:
                    bx = x_chunks[citem["idx"]]
                    ops = citem.get("ops", ())
                    for op in ops:
                        if op[0] in "hv" and op[1:] == "flip":
                            bx = torch.flip(bx, dims=(-1 if op[0] == "h" else -2,))
                        elif op[0] in "udrl" and op[1:] == "roll":
                            bx = torch.roll(
                                bx,
                                shifts=-1 if op[0] in "ul" else 1,
                                dims=-1 if op[0] in "rl" else -2,
                            )
                        else:
                            raise ValueError("bad op")
                    if hoverlap > 0 and cresult:
                        cresult[-1][..., -hoverlap:] = blend(
                            cresult[-1][..., -hoverlap:],
                            bx[..., :hoverlap],
                            blend_strength,
                        )
                        bx = bx[..., hoverlap:]
                    cresult.append(bx)
                cx = torch.cat(cresult, dim=-1)
                if voverlap > 0 and rresult:
                    rresult[-1][..., -voverlap:, :] = blend(
                        rresult[-1][..., -voverlap:, :],
                        cx[..., :voverlap, :],
                        blend_strength,
                    )
                    cx = cx[..., voverlap:, :]
                rresult.append(cx)
                del cresult
            bresult.append(torch.cat(rresult, dim=-2))
            del rresult
        del x_chunks
        return torch.cat(bresult, dim=0)

    @classmethod
    def go(
        cls,
        *,
        sampler,
        mode,
        merge_step,
        hoverlap,
        voverlap,
        blend_mode="lerp",
        blend_strength=0.5,
        sampler_after=None,
        advanced_plan="",
    ):
        if sampler_after is None:
            sampler_after = sampler
        merge_step -= 1

        if advanced_plan:
            plan = cls.parse_plan(advanced_plan)
            if not plan:
                plan = None
        else:
            plan = None

        blend = utils.BLENDING_MODES[blend_mode]

        def sampler_fun(
            model,
            x,
            sigmas,
            extra_args=None,
            **kwargs: dict,
        ):
            nonlocal plan, merge_step
            if plan is None:
                if mode == "vertical":
                    plan = (
                        {
                            "batch": (
                                tuple(({"idx": bidx},) for bidx in range(x.shape[0])),
                            ),
                        },
                    )
                elif mode == "horizontal":
                    plan = (
                        {
                            "batch": (
                                (tuple({"idx": bidx} for bidx in range(x.shape[0])),),
                            ),
                        },
                    )
                else:
                    raise ValueError("Bad mode")
                plan[0]["step"] = merge_step
            cls.dump_plan(plan)
            pi = plan[0]
            merge_step = pi["step"]
            extra_args = extra_args.copy() if extra_args is not None else {}
            before_sigmas = sigmas[: merge_step + 1]
            after_sigmas = sigmas[merge_step:]
            print("\nBEFORE:", before_sigmas)
            print("AFTER:", after_sigmas)
            x = sampler.sampler_function(
                model,
                x,
                before_sigmas,
                extra_args=extra_args,
                **kwargs,
                **sampler.extra_options,
            )
            x = cls.execute_plan_item(
                x,
                pi,
                hoverlap=hoverlap,
                voverlap=voverlap,
                blend=blend,
                blend_strength=blend_strength,
            )
            if after_sigmas.numel() < 2:
                return x
            seed = extra_args.get("seed")
            if seed is not None:
                extra_args["seed"] = seed + 1000
            return sampler_after.sampler_function(
                model,
                x,
                after_sigmas,
                extra_args=extra_args,
                **kwargs,
                **sampler_after.extra_options,
            )

        return (KSAMPLER(sampler_fun, inpaint_options=sampler.inpaint_options),)


NODE_CLASS_MAPPINGS = {
    "BatchMergeSampler": BatchMergeSampler,
}
