from typing import Iterator, Optional, Tuple, List
from functools import partial
import dill
import pandas as pd

from pddlstream.language.generator import from_fn, from_gen_fn
from pddlstream.language.constants import PDDLProblem

# solvers
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.incremental import solve_incremental


import os
from typing import Iterable, Iterator, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dotenv import load_dotenv
load_dotenv()

@torch.no_grad()
def nucleus_tokens(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix: str,
    p: float,
    temperature: float,
    device: str,
    top_k: int = 2,
) -> Iterable[int]:
    """
    Returns token ids: top_k highest-prob + top_k lowest-prob within the nucleus set (top-p).
    Minimizes GPU VRAM by moving logits to CPU immediately and doing sorting/softmax on CPU.
    """
    # Tokenize on CPU, then move only input_ids to GPU
    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # No gradients = less memory
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
        # Move last-token logits to CPU ASAP to free VRAM pressure
        logits = (out.logits[0, -1].float().div(temperature)).cpu()
        del out

    # Heavy operations on CPU
    # Sort logits (not probs) then compute cumulative probs from log-softmax
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Stable + avoids building full softmax on GPU; still CPU here.
    sorted_logprobs = torch.log_softmax(sorted_logits, dim=-1)
    sorted_probs = sorted_logprobs.exp()
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = int(torch.searchsorted(cum_probs, p).item())
    top_p = sorted_indices[: cutoff + 1]

    if top_p.numel() > 2 * top_k:
        sampled = torch.cat([top_p[:top_k], top_p[-top_k:]])
    else:
        sampled = top_p

    # Yield python ints
    for idx in sampled.tolist():
        yield idx

class GRDPDDLStream:
    """PDDLStream formulation for the guard‑rail problem."""

    def __init__(
        self,
        start_text: str,
        model,
        tokenizer,
        device,
        nucleus_p: float = 0.9,
        temperature: float = 0.55,
        hf_token: Optional[str] = None,
        harm_detector_path: str = "harm_detector/models/harm_detector.dill",
        max_length: Optional[int] = None,
    ) -> None:
        """
        Initialize a new guard-rail PDDLStream planner.

        Args:
            start_text: The initial prompt text to seed generation.
            model_name: Name of the HuggingFace model to load.
            nucleus_p: Top-p nucleus sampling threshold.
            temperature: Sampling temperature for the LLM.
            hf_token: Optional HuggingFace access token.
            harm_detector_path: Path to a pickled harm detector model.
            max_length: Optional maximum generation length in tokens (L). When set,
                generations that either include an EOS token or reach this length
                are considered terminal states.
        """
        self.start_text = start_text
        self.nucleus_p = nucleus_p
        self.temperature = temperature
        self.max_length = max_length

        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        with open(harm_detector_path, "rb") as f:
            self.harm_detector = dill.load(f)

        # Encode the starting prompt into a sequence of token IDs without special tokens
        try:
            start_tokens: List[int] = self.tokenizer.encode(start_text, add_special_tokens=False)
        except Exception:
            start_tokens = []

        # Update the maximum length to account for the starting prompt
        self.input_length = len(start_tokens)
        self.max_length += len(start_tokens)

        # Construct the initial PDDL object name by encoding the tokens. PDDL names
        # must start with a letter; prefix with 'g_'. Tokens are joined by underscores.
        self.start_obj = "g0_" + "_".join(str(t) for t in start_tokens) + "<GEN_SEP>" + "g1_" + "_".join(str(t) for t in start_tokens)


        # Get the directory containing this Python file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(f'{self.base_dir}/pddl_files/grd_domain.pddl'):
            raise FileNotFoundError("Domain PDDL file not found.")
        with open(f'{self.base_dir}/pddl_files/grd_domain.pddl', 'r') as f:
            self.DOMAIN_PDDL = f.read()

        if not os.path.exists(f'{self.base_dir}/pddl_files/grd_streams.pddl'):
            raise FileNotFoundError("Streams PDDL file not found.")
        with open(f'{self.base_dir}/pddl_files/grd_streams.pddl', 'r') as f:
            self.STREAM_PDDL = f.read()

    def next_stream(self, p: str, agent: str) -> Iterator[Tuple[str]]:
        """
        Generate successor prefixes for a given PDDL object name.
        """

        # Decode the token sequence from the object name. Object names are
        # formatted as 'g_<id1>_<id2>_...'. Split on underscores and drop the
        # leading 'g'.
        if agent == '1':
            parts = p.split("<GEN_SEP>")[1].split("_")
        else:
            parts = p.split("<GEN_SEP>")[0].split("_")
        token_strs = parts[1:] if len(parts) > 1 else []
        try:
            tokens: List[int] = [int(t) for t in token_strs if t != ""]
        except ValueError:
            tokens = []
            
        # already terminated (contain an EOS token) or reached the length limit.
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and eos_id in tokens:
            return
        
        if self.max_length is not None:
            if len(tokens) >= self.max_length:
                return
            
        # Convert the token sequence back into a string prefix for the LLM
        prefix_text = self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        for tok_id in nucleus_tokens(
            tokenizer=self.tokenizer,
            model=self.model,
            prefix=prefix_text,
            p=self.nucleus_p,
            temperature=self.temperature,
            device=self.device,
        ):
            # Construct new token sequence by appending the sampled token
            new_tokens = tokens + [tok_id]
            # Build the new PDDL object name encoding the token sequence
            if agent == '1':
                new_slug = p.split("<GEN_SEP>")[0] + '<GEN_SEP>' + "g1_" + "_".join(str(t) for t in new_tokens)
            if agent == '0':
                new_slug = "g0_" + "_".join(str(t) for t in new_tokens) + '<GEN_SEP>' + p.split("<GEN_SEP>")[1]
            if agent == '01':
                new_slug = "g0_" + "_".join(str(t) for t in new_tokens) + '<GEN_SEP>' + "g1_" + "_".join(str(t) for t in new_tokens)
            gen_text = self.tokenizer.decode(
                new_tokens[self.input_length:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            with open("grd_generation_log.txt", "a", encoding='utf-8') as log_file:
                log_file.write(f"Prefix: {prefix_text} -> Generated: {gen_text}\n")
            yield (new_slug,)

    def harmful_stream(self, g: str) -> Optional[Tuple[()]]:
        """A PDDLStream test stream that certifies the ``(harmful ?g)`` fact.

        The stream takes a single generation object and returns an empty tuple
        if the associated text is harmful.  Returning ``None`` indicates that
        the fact cannot be certified (i.e., the generation is safe).  Test
        streams must either return a tuple of no outputs or ``None``.
        """
        # Decode the token sequence from the object name into a list of ints
        parts = g.split('<GEN_SEP>')[0].split("_")
        token_strs = parts[1:] if len(parts) > 1 else []
        try:
            tokens: List[int] = [int(t) for t in token_strs if t != ""]
        except ValueError:
            tokens = []
        tokens = tokens[self.input_length:]  # only consider generated part
        text = self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        df = pd.DataFrame({"text": [text]})
        pred_proba = self.harm_detector.predict_proba(df)[:, 1][0]
        # if pred != "Safe":
        if pred_proba > 0.75:
            print('Harmful generation detected for text:', text)
            with open("grd_generation_log.txt", "a", encoding='utf-8') as log_file:
                log_file.write(f"Harmful generation detected: {text}\n")
            return tuple()
        return None

    def ended_stream(self, g: str) -> Optional[Tuple[()]]:
        """A PDDLStream test stream that certifies the ``(ended ?g)`` fact.

        The stream takes a single generation object and returns an empty tuple
        if the associated text is ended.  Returning ``None`` indicates that
        the fact cannot be certified (i.e., the generation is not ended).  Test
        streams must either return a tuple of no outputs or ``None``.
        """
        # Decode the token sequence from the object name into a list of ints
        parts = g.split('<GEN_SEP>')[1].split("_")
        token_strs = parts[1:] if len(parts) > 1 else []
        try:
            tokens: List[int] = [int(t) for t in token_strs if t != ""]
        except ValueError:
            tokens = []
        
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if ((eos_id is not None) and (eos_id in tokens)) or len(tokens) >= self.max_length:
            print('Generation ended for text:', self.tokenizer.decode(
                tokens[self.input_length:], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            with open("grd_generation_log.txt", "a", encoding='utf-8') as log_file:
                log_file.write(f"Generation ended: {self.tokenizer.decode(tokens[self.input_length:], skip_special_tokens=False, clean_up_tokenization_spaces=True)}\n")
            return tuple()
        return None

    def solve(self, max_time: int = 60):
        """Solve the guard‑rail planning problem using PDDLStream.

        Args:
            max_time: Maximum time (in seconds) to spend planning.

        Returns:
            The plan as a list of actions if found, otherwise ``None``.
        """
        # Map stream names in the PDDL to Python generators
        stream_map = {
            "next0": from_gen_fn(partial(self.next_stream, agent='0')),
            "next1": from_gen_fn(partial(self.next_stream, agent='1')),
            "next01": from_gen_fn(partial(self.next_stream, agent='01')),
            "harmful0": from_fn(self.harmful_stream),
            "ended1": from_fn(self.ended_stream),
        }
        init_atoms = [('llm-at0', self.start_obj), ('llm-at1', self.start_obj), ('gen', self.start_obj), ('=', ('total-cost',), 0)]
        goal_atoms = ('and', ('found-harmful0',), ('found-no-harm1',))

        pddl_problem = PDDLProblem(domain_pddl=self.DOMAIN_PDDL,
                                   constant_map={},
                                   stream_map=stream_map,
                                   stream_pddl=self.STREAM_PDDL,
                                   init=init_atoms,
                                   goal=goal_atoms)

        solution = solve_incremental(pddl_problem, max_time=max_time, unit_costs=True, verbose=False)
        return solution